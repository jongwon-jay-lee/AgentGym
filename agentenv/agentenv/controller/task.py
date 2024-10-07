from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, TypedDict

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel
from transformers import GenerationConfig, PreTrainedModel, PreTrainedTokenizerBase
from transformers.generation.utils import GenerateOutput

from agentenv.controller import BaseEnvClient

ConversationMessage = TypedDict(
    "ConversationMessage", {"from": str, "loss": Optional[bool], "value": str}
)


@dataclass
class ExperienceOutput:
    conversation: list[ConversationMessage]
    reward: float
    text: str
    seq_ids: list[int]
    attention_mask: list[int]
    action_mask: list[int]


TokenizedConversationOutput = TypedDict(
    "TokenizedConversationOutput",
    {
        "text": str,
        "input_ids": list[int],
        "action_mask": list[int],
    },
)


class BaseTask:
    env_client_cls: Callable
    env_name: str

    def __init__(
        self,
        client_args: Mapping[str, Any],
        n_clients: int = 1,
    ) -> None:
        """
        Initializes the Task object.

        Args:
            client_args (Mapping[str, Any]): A mapping of client arguments.
            n_clients (int, optional): The number of clients. Defaults to 1. Larger than 1 for batch generation. Batch generation is not implemented yet.
        """
        if self.env_client_cls is None or self.env_name is None:
            raise NotImplementedError
        self.clients = [self.env_client_cls(**client_args) for _ in range(n_clients)]
        self.len = len(self.clients[0])

    def _tokenize_conversation_one(
        self,
        message: ConversationMessage,
        tokenizer: PreTrainedTokenizerBase,
    ) -> TokenizedConversationOutput:
        """
        This function applied Llama Chat template on the given vicuna-styled conversation message.
        You can provide your own _tokenize_conversation_one to adapt to your own task.
        """
        if message["from"] == "human":
            text = f"<s>[INST] {message['value']} [/INST]"
            input_ids = tokenizer.encode(text, add_special_tokens=False)
        else:
            text = f"{message['value']}</s>"
            input_ids = tokenizer.encode(text, add_special_tokens=False)
            text = f" {text}"
        if message["loss"]:
            action_mask = [1] * len(input_ids)
        else:
            action_mask = [0] * len(input_ids)

        return TokenizedConversationOutput(
            {
                "text": text,
                "input_ids": input_ids,
                "action_mask": action_mask,
            }
        )

    def _tokenize_conversation(
        self,
        conversation: list[ConversationMessage],
        tokenizer: PreTrainedTokenizerBase,
    ) -> TokenizedConversationOutput:
        text = ""
        input_ids = []
        action_mask = []

        for message in conversation:
            message_out = self._tokenize_conversation_one(message, tokenizer)
            text += message_out["text"]
            input_ids += message_out["input_ids"]
            action_mask += message_out["action_mask"]

        return TokenizedConversationOutput(
            {
                "text": text,
                "input_ids": input_ids,
                "action_mask": action_mask,
            }
        )

    def _generate_experience_one(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        client: BaseEnvClient,
        idx: int,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> ExperienceOutput:
        client.reset(idx)
        reward = 0.0
        done = False
        state = client.observe()
        conversation = list(client.conversation_start)
        conversation.append(
            ConversationMessage({"from": "human", "loss": None, "value": state})
        )
        conversation_tokenized = self._tokenize_conversation(conversation, tokenizer)
        rounds = 0

        while not done:
            input_length = len(conversation_tokenized["input_ids"])
            # if input_length exceeds 4096, break
            if input_length >= model.config.max_length:
                break
            output = model.generate(
                torch.tensor(
                    [conversation_tokenized["input_ids"]], device=model.device
                ),
                generation_config=generation_config,
            )
            if isinstance(output, GenerateOutput):
                output = output.sequences

            generated_tokens = output[0][input_length:].cpu().numpy().tolist()
            if generated_tokens[-1] != tokenizer.eos_token_id:
                generated_tokens += [tokenizer.eos_token_id]

            generated_text = tokenizer.decode(generated_tokens)
            conversation_tokenized["text"] += f" {generated_text}"
            conversation_tokenized["input_ids"] += generated_tokens
            conversation_tokenized["action_mask"] += [1] * len(generated_tokens)

            generated_text = generated_text[
                : -len(tokenizer.eos_token)
            ]  # not endswith eos_token
            conversation.append(
                ConversationMessage(
                    {"from": "gpt", "loss": True, "value": generated_text}
                )
            )

            step_output = client.step(generated_text)
            state, reward, done = (
                step_output.state,
                step_output.reward,
                step_output.done,
            )
            env_message = ConversationMessage(
                {"from": "human", "loss": None, "value": state}
            )
            env_message_tokenized = self._tokenize_conversation_one(
                env_message, tokenizer
            )

            conversation.append(env_message)
            conversation_tokenized["text"] += env_message_tokenized["text"]
            conversation_tokenized["input_ids"] += env_message_tokenized["input_ids"]
            conversation_tokenized["action_mask"] += env_message_tokenized[
                "action_mask"
            ]

            rounds += 1
            if max_rounds is not None and rounds >= max_rounds:
                break

        return ExperienceOutput(
            conversation=conversation,
            reward=reward,
            text=conversation_tokenized["text"],
            seq_ids=conversation_tokenized["input_ids"],
            attention_mask=[1] * len(conversation_tokenized["input_ids"]),
            action_mask=conversation_tokenized["action_mask"],
        )

    def impl_generate_experience_batch(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        clients: BaseEnvClient,
        idxs: list[int],
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
        accelerator =None
    ):
        tokenizer.padding_side = "left"
        # check the pad_token exists or not
        if tokenizer.pad_token_id is None:
            print("Use EOS token as PAD")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            model.config.pad_token_id = model.config.eos_token_id

        bsz = len(idxs)
        
        curr_process_idx = accelerator.process_index

        client_start_idx = 0
        client_end_idx = bsz
        
        # Reset all envs, rewards, and dones
        states = []
        for c_idx in range(client_start_idx, client_end_idx):
            clients[c_idx].reset(idxs[c_idx - client_start_idx])
            state = clients[c_idx].observe()
            states.append(state)
        rewards = [0.0] * bsz
        dones = [False] * bsz

        conversation_tokenized_lst, conversations = [], []
        for c_idx in range(client_start_idx, client_end_idx):
            conversation = list(clients[c_idx].conversation_start)
            conversation.append(
                ConversationMessage({
                    "from": "human",
                    "loss": None,
                    "value": states[c_idx - client_start_idx]
                })
            )
            conversations.append(conversation)
            conversation_tokenized = self._tokenize_conversation(conversation, tokenizer)
            conversation_tokenized_lst.append(conversation_tokenized)
        rounds = [0] * bsz

        while True:
            if sum(dones) == bsz:
                break
            # Setup the max len for current batch
            max_len = -1
            input_length_lst = []
            for d_idx in range(len(dones)):
                if dones[d_idx]:
                    continue
                input_length = len(conversation_tokenized_lst[d_idx]['input_ids'])
                if input_length >= model.config.max_length:
                    dones[d_idx] = True
                    continue
                input_length_lst.apppend(input_length)
                max_len = max(max_len, input_length)
            
            # Check weather all trajectory are complete or not
            if sum(dones) == bsz:
                break
            
            # Prepare input tensors for batch run
            all_input_ids = []
            all_attn_masks = []
            curr_lens = []
            for d_idx in range(len(dones)):
                if dones[d_idx]:
                    continue
                curr_len = len(conversation_tokenized_lst[d_idx]['input_ids'])
                curr_lens.append(curr_len)
                
                try:
                    conversation_tokenized_lst[d_idx]['input_ids'] = torch.cat([
                        torch.IntTensor([tokenizer.pad_token_id] * (max_len - curr_len)), 
                        torch.tensor(conversation_tokenized_lst[d_idx]['input_ids'])
                    ]).to(model.device)
                    attn_masks = torch.cat([
                        torch.IntTensor([0] * (max_len - curr_len)),
                        torch.ones(curr_len, dtype=torch.long)
                    ]).to(model.device)
                except:
                    raise(f"ERROR on idx: {d_idx}, Shape: {torch.tensor(conversation_tokenized_lst[d_idx]['input_ids']).shape}")
                
                all_input_ids.append(conversation_tokenized_lst[d_idx]['input_ids'])
                all_attn_masks.append(attn_masks)

            # Stacking tensors as batch-wise
            all_input_ids = torch.stack(all_input_ids)
            all_attn_masks = torch.stack(all_attn_masks)

            output = model.generate(
                input_ids = all_input_ids,
                attention_mask = all_attn_masks,
                generation_config = generation_config,
                eos_token_id = tokenizer.eos_token_id,
                pad_token_id = tokenizer.pad_token_id
            )

            if isinstance(output, GenerateOutput):
                output = output.sequences
            
            # Decode batch-generated tokens
            ord = -1
            all_generated_tokens = []
            for d_idx in range(len(dones)):
                if dones[d_idx]:
                    continue
                ord += 1
                generated_tokens = output[ord][max_len:].cpu().numpy().tolist()
                # <UNK> is padded if a model/tokenizer sets either unk_token or eos_token as pad_token.
                if generated_tokens[-1] != tokenizer.unk_token_id and generated_tokens[-1] != tokenizer.eos_token_id:
                    generated_tokens += [tokenizer.eos_token_id]
                all_generated_tokens.append(generated_tokens)

            all_generated_texts = tokenizer.batch_decode(all_generated_tokens)

            # Adjust the max len for current batch
            ord = -1
            for d_idx in range(len(dones)):
                if dones[d_idx]:
                    continue
                ord += 1
                curr_len = curr_lens[ord]

                # Handling special tokens (UNK, PAD)
                all_generated_texts[ord] = all_generated_texts[ord].replace(tokenizer.unk_token, "").replace(tokenizer.pad_token, "")
                valid_idxs = (np.array(all_generated_tokens[ord]) != tokenizer.unk_token_id) & (np.array(all_generated_tokens[ord]) != tokenizer.pad_token_id)
                all_generated_tokens[ord] = np.array(all_generated_tokens[ord])[valid_idxs].tolist()

                conversation_tokenized_lst[d_idx]['input_ids'] = conversation_tokenized_lst[d_idx]['input_ids'][-curr_len:].cpu().numpy().tolist()

                # Add the very last EOS token at the end of the sequences
                if tokenizer.eos_token_id == tokenizer.pad_token_id:
                    all_generated_texts[ord] += tokenizer.eos_token
                    all_generated_tokens[ord] += [tokenizer.eos_token_id]
                
                conversation_tokenized_lst[d_idx]["text"] += f" {all_generated_texts[ord]}"
                conversation_tokenized_lst[d_idx]["input_ids"] += all_generated_tokens[ord]
                conversation_tokenized_lst[d_idx]["action_mask"] += [1] * len(all_generated_tokens[ord])
                # Drop the textified eos_token for upcoming generation
                all_generated_texts[ord] = all_generated_texts[ord][:-len(tokenizer.eos_token)]
                conversations[d_idx].append(
                    ConversationMessage(
                        {"from": "gpt", "loss": True, "value": all_generated_texts[ord]}
                    )
                )
                # Step ENV
                step_output = clients[client_start_idx + d_idx].step(all_generated_texts[ord])
                state, reward, done = (
                    step_output.state, 
                    step_output.reward,
                    step_output.done,
                )
                rewards[d_idx] = reward
                dones[d_idx] = done
                env_message = ConversationMessage(
                    {"from": "human", "loss": None, "value": state}
                )
                env_message_tokenized = self._tokenize_conversation_one(env_message, tokenizer)
                conversations[d_idx].append(env_message)
                conversation_tokenized_lst[d_idx]["text"] += env_message_tokenized["text"]
                conversation_tokenized_lst[d_idx]["input_ids"] += env_message_tokenized["input_ids"]
                conversation_tokenized_lst[d_idx]["action_mask"] += env_message_tokenized["action_mask"]
                rounds[d_idx] += 1

                if max_rounds is not None and rounds[d_idx] >= max_rounds:
                    dones[d_idx] = True

        results = []
        for d_idx in range(len(dones)):
            results.append(ExperienceOutput(
                conversation=conversations[d_idx],
                reward=rewards[d_idx],
                text=conversation_tokenized_lst[d_idx]["text"],
                seq_ids=conversation_tokenized_lst[d_idx]["input_ids"],
                attention_mask=[1] * len(conversation_tokenized_lst[d_idx]["input_ids"]),
                action_mask=conversation_tokenized_lst[d_idx]["action_mask"]
            ))
        return results

    def _generate_experience_batch(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        idxs: Sequence[int],
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
        accelerator = None,
    ) -> list[ExperienceOutput]:
        
        if len(self.clients) == 1:
            client = self.clients[0]
            result = [self._generate_experience_one(
                        model=model,
                        tokenizer=tokenizer,
                        client=client,
                        idx=idx,
                        generation_config=generation_config,
                        max_rounds=max_rounds,
                    ) for idx in idxs]
        else:
            result = self.impl_generate_experience_batch(
                model = model,
                tokenizer = tokenizer,
                clients = self.clients,
                idxs = idxs,
                generation_config = generation_config,
                max_rounds = max_rounds,
                accelerator = accelerator
            )
        return result

    def generate_experience(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        idxs: Sequence[int] | int,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> list[ExperienceOutput]:
        if isinstance(idxs, int):
            idxs = [idxs]

        if isinstance(model, DistributedDataParallel):
            model = model.module

        return self._generate_experience_batch(
            model=model,
            tokenizer=tokenizer,
            idxs=idxs,
            generation_config=generation_config,
            max_rounds=max_rounds,
        )
