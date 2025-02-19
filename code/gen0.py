# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from typing import Tuple
from enum import Enum
from dataclasses import dataclass

import colorama
import datetime
import random
import sys
import torch
import traceback
import transformers
import os
from tqdm import tqdm
from arguments import Arguments, simple_parse_args_string
from self_speculation.autoregressive_generator import AutoRegressiveGenerationStrategy
from self_speculation.generator_base import (
    GenerationConfig,
    GenerationResult,
    GenerationStrategy,
    HuggingfaceLlamaGenerator,
)
from self_speculation.self_speculation_generator import SelfSpeculativeGenerationStrategy
from self_speculation.speculative_streamer import SpeculativeTextStreamer


class StreamerType(str, Enum):
    NONE = "none"
    STANDARD = "standard"
    SPECULATIVE = "speculative"


@dataclass
class GenerateArguments:
    streamer: StreamerType = StreamerType.STANDARD


def setup(args: Arguments, device: str = "cuda"):
    backend_str = "cpu:gloo" if "cpu" in device else "cuda:nccl,cpu:gloo"
    torch.distributed.init_process_group(
        backend=backend_str, timeout=datetime.timedelta(hours=48)
    )
    rank = int(os.environ["LOCAL_RANK"])

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if rank != 0:
        # only run on rank 0, we don't support parallel inference yet
        exit()


def load_model_and_tokenizer(args: Arguments, device: str = "auto"):
    local_model_path: str = args.model

    # initialize model
    tokenizer = transformers.AutoTokenizer.from_pretrained(local_model_path)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        local_model_path,
        use_safetensors=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()

    return model, tokenizer


def main(args: Arguments, generate_arguments: GenerateArguments, generation_config: GenerationConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    setup(args, device=device)
    transformers.utils.logging.set_verbosity_error()
    model, tokenizer = load_model_and_tokenizer(args, device=device)

    streamer = None
    if generate_arguments.streamer == StreamerType.NONE:
        streamer = None
    elif generate_arguments.streamer == StreamerType.STANDARD:
        streamer = transformers.TextStreamer(tokenizer)
    elif generate_arguments.streamer == StreamerType.SPECULATIVE:
        streamer = SpeculativeTextStreamer(tokenizer)
    else:
        raise ValueError(f"Unsupported streamer type {generate_arguments.streamer}")

    if generation_config.generation_strategy == "autoregressive":
        generation_strategy: GenerationStrategy = AutoRegressiveGenerationStrategy()
    elif generation_config.generation_strategy == "self_speculative":
        generation_strategy: GenerationStrategy = SelfSpeculativeGenerationStrategy()
    else:
        raise Exception(
            f"Unsupported generation strategy: {generation_config.generation_strategy}"
        )

    # initialize generator
    generator = HuggingfaceLlamaGenerator(
        tokenizer=tokenizer, model=model, generation_strategy=generation_strategy
    )

    # Warmup
    warmup = 1
    for _ in range(warmup):
        model.generation_config.pad_token_id = tokenizer.eos_token_id
        model.generate(**tokenizer("This is a warmup prompt", return_tensors="pt").to(device), max_new_tokens=10)
# # #     #4 53-56 261-280
    questions = [
        "the cast of the movie the four seasons",
        "who wrote the song whiter shade of pale",
        "where are the remaining copies of the magna carta",
        "what are the numbers in the house now",
        "when did won't you be my neighbor come out"
        # "how many episodes in the walking dead season nine",
        # "what season of top chef did richard win",
        # "who did the united states fight against in world war ii",
        # "what are burger king chicken nuggets really made of",
        # "how long did it take odysseus to return home"
        # "what is it called when you write japanese with english letters",
        # "where does the path train go in new york",
        # "countries where english is first or native language",
        # "what are the three largest cities in usa",
        # "who won women's doubles at us open"
        # "who sang the music in a star is born",
        # "what are the three layers of the eyes",
        # "members of the black order in infinity war",
        # "top 10 teams with most premier league titles",
        # "who explained planetary motion by describing epicycles deferents and equants"
    ]
# # # #     #5 57-60
#     questions = [
#         # "who starred in the first star is born movie",
#         # "when did juice wrld first song come out",
#         # "who sponsored icc world cup outside england first time",
#         # "presidents that won popular vote but not election",
#         # "where is the maximum ozone depletion has been observed"
#         "who starred in the girl on the train",
#         "who were the original members of the beatles",
#         "where can salivary amylase be found in the body",
#         "university of minnesota college of biological sciences majors",
#         "when was the song we are the champions written"
#         # "who are the candidates running for florida governor",
#         # "what are the most populated cities in the us",
#         # "first stage of kohlberg's preconventional level of moral reasoning",
#         # "who's running for mayor in washington dc",
#         # "when was icc cricket world cup held outside england"
#         # "who was authored the book titled feminist dictionary",
#         # "what are the biggest car companies in the world",
#         # "when does my hero academia 2 heroes come out",
#         # "where was the movie the river runs through it filmed",
#         # "who sings you gotta lick it before you stick it"
#     ]

    output_dir_base = "nq/13b/nq_13"

    if not os.path.exists(output_dir_base):
        os.makedirs(output_dir_base)

    for idx, question in enumerate(questions, start=1):
        print(f"Processing question {idx}: {question}")
        prompt = f"Answering the following question with no other comment.\n Question: {question}\n Answer: "
        output_dir = os.path.join(output_dir_base, f"question_{idx+60}_7b")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 设置文件名
        exit_layer = generation_config.exit_layer  # 根据配置设置的 exit_layer
        output_filename = os.path.join(output_dir, f"answer_layer{exit_layer}.txt")

        with open(output_filename, "w") as output_file:
            for i in tqdm(range(50)):
                try:
                    # 构建 prompt
                    # 生成响应
                    response: GenerationResult = generator.generate(
                        prompt=prompt,
                        generation_config=generation_config,
                        streamer=None,
                    )

                    # 处理和清理生成的文本
                    text = response.decoded_prediction
                    if text.strip()[:2] in ['1.', '2.', 'A.']:
                        text = text.split(".")[1]
                    elif text.strip()[:2] == '1':
                        if "2)" in text:
                            text = text.split(")")[1][:-2].strip()
                        else:
                            text = text.split(")")[1].strip()
                    elif text.strip()[:5] in ["A: 1", "A: 2"]:
                        text = text.split(".")[1]

                    text = text.split("\n")[0]
                    # 打印生成的文本
                    print(f"{i}-th Generated Text for question '{question}':", text)

                    # 写入文件
                    output_file.write(f"Generated Response {i + 1}:\n")
                    output_file.write(text + "\n")
                    output_file.write("-" * 50 + "\n")  # 分隔线

                    # 刷新文件内容
                    output_file.flush()

                except Exception as e:
                    print(colorama.Style.RESETALL)
                    traceback.print_exc()
                    raise e

        print(f"Generated answers for question '{question}' are saved in {output_filename}")

    print("All questions processed.")

def process_cli_arguments() -> Tuple[Arguments, GenerateArguments, GenerationConfig]:
    parser = transformers.HfArgumentParser((Arguments, GenerateArguments, GenerationConfig))
    (
        general_arguments,
        generate_arguments,
        generation_config,
        _remaining,
    ) = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    if general_arguments.model_args:
        general_arguments.model_args = simple_parse_args_string(general_arguments.model_args)
    else:
        general_arguments.model_args = {}

    return general_arguments, generate_arguments, generation_config


if __name__ == "__main__":
    args, benchmark_arguments, generation_config = process_cli_arguments()
    main(args, benchmark_arguments, generation_config)
