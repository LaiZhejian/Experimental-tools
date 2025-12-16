#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import asyncio
import hashlib
import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable

import openai
import yaml
from tqdm import tqdm

# =======================
# DEFAULT CONFIG (YAML 覆盖它)
# =======================
CONFIG: Dict[str, Any] = {
    "llm": {"name": "JsonLLM"},

    "api": {
        "base_url": "http://127.0.0.1:10086/v1",
        "api_key": "EMPTY",
        "model": "default",
        "timeout": 3600,
        "extra_body": None,
    },

    "runner": {
        "sem_num": 256,
        "batch_size": 16384,
        "write_every_n": 1,
        "count_total_jsonl": False,  # jsonl 是否扫 total（大文件建议 false）
    },

    "retry": {
        "num_retry": 4,
        "base_delay": 1.0,
        "max_delay": 60.0,
        "retry_on_status": [429, 500, 502, 503, 504],
        "break_on_status": [400, 401, 403, 404, 422, 450],

    },

    "generation": {
        "sampling_params": {"temperature": 0.6, "top_p": 0.95, "max_tokens": 4096, "n": 4},
        "beam_search": None,
        "resample_rounds": 3,  # parse/format 失败：补齐最多几轮
        "record_dropped_parse_errors": True,
        "max_dropped_errors": 3,  # 限制 parse_errors 长度
        "require_exact_n": False,  # 不够 n 标记 incomplete=true
    },

    "data": {
        "prompt_key": "prompt",
        "id_key": "generate_id",
        "id_mode": "index",  # "index" | "hash"
        "hash_role_content": True,  # hash 时只 hash role+content

        "output_key": "generated_texts",
        "parsed_key": "parsed",
        "parse_error_key": "parse_errors",
        "incomplete_key": "incomplete",
        "error_key": "error",  # one() try/except 写入

        "best_index_key": "best_index",
        "best_text_key": "best_text",
        "best_parsed_key": "best_parsed",
    },

    # 处理层：放到基类 LLM
    "processing": {
        "strip_think": True,  # 所有 LLM 统一先去 <think>…</think>
    },

    "json_llm": {
        "extract_json_fence": True,
        "require_dict_or_list": True,
    },

    "resume": {"enabled": True},
    "best_of_n": {"enabled": True},
    "log_level": "INFO",
}


# =======================
# utils
# =======================
def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(out.get(k), dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f) or {}
    if not isinstance(obj, dict):
        raise ValueError("YAML config must be a dict")
    return obj


def count_jsonl_lines(path: str) -> int:
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def iter_jsonl_batches(path: str, batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def iter_list_batches(items: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def canonical_for_hash(prompt: Any, role_content_only: bool) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, list) and role_content_only:
        slim = []
        for m in prompt:
            if isinstance(m, dict):
                slim.append({"role": m.get("role"), "content": m.get("content")})
        return json.dumps(slim, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return json.dumps(prompt, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def make_hash_id(item: Dict[str, Any], prompt_key: str, role_content_only: bool) -> str:
    s = canonical_for_hash(item.get(prompt_key), role_content_only)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def stream_load_done_ids(output_path: str, id_key: str) -> set:
    done = set()
    if not os.path.exists(output_path):
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                it = json.loads(line)
            except Exception:
                continue
            gid = it.get(id_key)
            if gid is not None:
                done.add(str(gid))
    return done


# =======================
# LLMFactory
# =======================
class LLMFactory:
    reg: Dict[str, Any] = {}

    @classmethod
    def register(cls, name: Optional[str] = None):
        def deco(c):
            cls.reg[name or c.__name__] = c
            return c

        return deco

    @classmethod
    def create(cls, name: str, cfg: Dict[str, Any]):
        if name not in cls.reg:
            raise ValueError(f"Unknown LLM: {name}. Available={list(cls.reg)}")
        return cls.reg[name](cfg)


# =======================
# Base LLM: 统一的 retry + resample 流程（子类只管 parse/format）
# =======================
@LLMFactory.register("LLM")
class LLM:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        api = cfg["api"]
        self.client = openai.AsyncClient(
            base_url=api["base_url"],
            api_key=api["api_key"],
            timeout=api["timeout"],
        )

    # 基类 post_process：只做通用预处理；子类在此基础上再 parse
    def post_process(self, text: str) -> str:
        if self.cfg.get("processing", {}).get("strip_think", False):
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()

    async def format_check(self, parsed: Any) -> bool:
        return True

    async def _call_api(self, prompt: Union[str, List[Dict[str, Any]]], gen_params: Dict[str, Any]) -> List[str]:
        api = self.cfg["api"]
        extra_body = api.get("extra_body")
        messages = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
        kwargs = dict(gen_params)
        if extra_body is not None:
            kwargs["extra_body"] = extra_body
        comp = await self.client.chat.completions.create(model=api["model"], messages=messages, **kwargs)
        return [(c.message.content or "") for c in comp.choices]

    async def generate(
            self, prompt: Union[str, List[Dict[str, Any]]], gen_params: Dict[str, Any]
    ) -> Tuple[List[str], List[Any], List[str]]:
        """
        目标：尽量凑够 n 个“parse+format 成功”的样本。
        - HTTP/网络级：retry.num_retry + backoff
        - parse/format 失败：通过 resample_rounds 做局部补齐（请求更多样本）
        返回：kept_texts, kept_parsed, dropped_errors（dropped_errors 已限长）
        """
        rc = self.cfg["retry"]
        retry_on = set(rc["retry_on_status"])
        break_on = set(rc["break_on_status"])
        delay, max_delay = float(rc["base_delay"]), float(rc["max_delay"])

        gc = self.cfg["generation"]
        target_n = int(gen_params.get("n", 1))
        rounds_total = 1 + max(0, int(gc.get("resample_rounds", 0)))
        max_errs = int(gc.get("max_dropped_errors", 0))

        kept_texts: List[str] = []
        kept_parsed: List[Any] = []
        dropped_errors: List[str] = []

        async def parse_and_keep(texts: List[str]):
            nonlocal dropped_errors
            for raw in texts:
                try:
                    # 子类 post_process 可能返回 parsed；基类默认只返回 str
                    parsed = self.post_process(raw)
                    if isinstance(parsed, str):
                        # 基类默认不 parse -> 当作失败（避免把原始文本当“成功样本”）
                        raise ValueError("post_process_not_implemented_in_subclass")
                    if not await self.format_check(parsed):
                        raise ValueError("format_check_failed")
                    kept_texts.append(raw)
                    kept_parsed.append(parsed)
                except Exception as e:
                    if max_errs > 0 and len(dropped_errors) >= max_errs:
                        continue
                    dropped_errors.append(str(e))

        for _round in range(rounds_total):
            need = target_n - len(kept_texts)
            if need <= 0:
                break

            cur_params = dict(gen_params)
            cur_params["n"] = need

            # HTTP/网络级重试
            cur_delay = delay
            for _ in range(int(rc["num_retry"])):
                try:
                    texts = await self._call_api(prompt, cur_params)
                    await parse_and_keep(texts)
                    break
                except Exception as e:
                    code = getattr(e, "status_code", None)
                    if code in break_on:
                        raise
                    if code is None or code in retry_on:
                        await asyncio.sleep(cur_delay)
                        cur_delay = min(max_delay, cur_delay * 2)
                        continue
                    raise

        return kept_texts, kept_parsed, dropped_errors


@LLMFactory.register("RawTextLLM")
class RawTextLLM(LLM):
    """
    不做结构化解析：保留清洗后的纯文本作为 parsed（用 dict 包一层，避免被基类当作未实现）。
    - 依然继承基类的：strip_think、HTTP retry、resample 补齐、format_check 流程
    - parsed: {"text": "<cleaned_text>"}  (始终是 dict，方便下游统一处理)
    """

    def post_process(self, text: str) -> Any:
        cleaned = super().post_process(text)
        return {"text": cleaned}

    async def format_check(self, parsed: Any) -> bool:
        return isinstance(parsed, dict) and isinstance(parsed.get("text"), str)


@LLMFactory.register("JsonLLM")
class JsonLLM(LLM):
    def post_process(self, text: str) -> Any:
        text = super().post_process(text)
        if self.cfg["json_llm"].get("extract_json_fence", True):
            m = re.findall(r"```json(.*?)```", text, re.DOTALL | re.IGNORECASE)
            if m:
                text = m[-1].strip()
        return json.loads(text)

    async def format_check(self, parsed: Any) -> bool:
        if not self.cfg["json_llm"].get("require_dict_or_list", True):
            return True
        return isinstance(parsed, (dict, list))

    # select_best 默认不在基类实现；需要就子类提供
    def select_best(self, texts: List[str], parsed: List[Any], item: Dict[str, Any]) -> int:
        return 0 if texts else -1


# =======================
# runner：one() 内 try/except；select_best 仅子类实现时调用
# =======================
async def run(cfg: Dict[str, Any], input_path: str, output_path: str):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    llm = LLMFactory.create(cfg["llm"]["name"], cfg)
    gen_params = cfg["generation"].get("sampling_params") or cfg["generation"].get("beam_search")
    if not isinstance(gen_params, dict):
        raise ValueError("generation params must be dict")

    sem = asyncio.Semaphore(int(cfg["runner"]["sem_num"]))
    bs = int(cfg["runner"]["batch_size"])
    write_every = int(cfg["runner"]["write_every_n"])

    d = cfg["data"]
    pk, id_key = d["prompt_key"], d["id_key"]
    out_k, parsed_k, err_k = d["output_key"], d["parsed_key"], d["parse_error_key"]
    incomplete_k, error_k = d["incomplete_key"], d["error_key"]

    best_enabled = bool(cfg.get("best_of_n", {}).get("enabled", True))
    has_best = best_enabled and hasattr(llm, "select_best") and callable(getattr(llm, "select_best"))

    # total & batches
    total: Optional[int] = None
    batches: Iterable[List[Dict[str, Any]]]

    if input_path.endswith(".jsonl"):
        if cfg["runner"].get("count_total_jsonl", False):
            total = count_jsonl_lines(input_path)
        batches = iter_jsonl_batches(input_path, bs)
    elif input_path.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if not isinstance(obj, list):
            raise ValueError(".json must be a list")
        total = len(obj)
        batches = iter_list_batches(obj, bs)
    elif input_path.endswith(".parquet"):
        try:
            import pandas as pd  # type: ignore
        except ImportError as e:
            raise ImportError("Reading parquet requires pandas+pyarrow") from e
        df = pd.read_parquet(input_path)
        total = int(df.shape[0])
        batches = iter_list_batches(df.to_dict(orient="records"), bs)
    else:
        raise ValueError("input must be .jsonl/.json/.parquet")

    done_ids = set()
    if cfg.get("resume", {}).get("enabled", True) and os.path.exists(output_path):
        done_ids = stream_load_done_ids(output_path, id_key)

    id_mode = d.get("id_mode", "index")
    role_content_only = bool(d.get("hash_role_content", True))
    require_exact_n = bool(cfg["generation"].get("require_exact_n", False))
    target_n = int(gen_params.get("n", 1))

    buffer: List[Dict[str, Any]] = []
    global_index = 0

    async def one(item: Dict[str, Any], idx: int) -> Optional[Dict[str, Any]]:
        gid = make_hash_id(item, pk, role_content_only) if id_mode == "hash" else str(idx)
        if gid in done_ids:
            return None

        async with sem:
            out = dict(item)
            out[id_key] = gid
            try:
                if pk not in item:
                    raise KeyError(f"missing {pk}")

                texts, parsed, dropped = await llm.generate(item[pk], gen_params)  # type: ignore
                out[out_k] = texts
                out[parsed_k] = parsed
                out[err_k] = dropped if cfg["generation"].get("record_dropped_parse_errors", True) else []
                out[incomplete_k] = bool(require_exact_n and (len(texts) < target_n))

                if has_best and texts:
                    bi = llm.select_best(texts, parsed, out)  # type: ignore
                    bi = bi if isinstance(bi, int) and 0 <= bi < len(texts) else 0
                    out[d["best_index_key"]] = bi
                    out[d["best_text_key"]] = texts[bi]
                    out[d["best_parsed_key"]] = parsed[bi]

                out[error_k] = None
            except Exception as e:
                out[out_k] = []
                out[parsed_k] = []
                out[err_k] = []
                out[incomplete_k] = True
                out[error_k] = str(e)
            return out

    with open(output_path, "a", encoding="utf-8") as fout:
        with tqdm(total=total, desc="Inferring", unit="sample") as pbar:
            for batch in batches:
                indexed = []
                for it in batch:
                    indexed.append((it, global_index))
                    global_index += 1

                tasks = [asyncio.create_task(one(it, idx)) for it, idx in indexed]
                for fut in asyncio.as_completed(tasks):
                    res = await fut
                    pbar.update(1)
                    if res is None:
                        continue
                    buffer.append(res)
                    if len(buffer) >= write_every:
                        fout.write("\n".join(json.dumps(x, ensure_ascii=False) for x in buffer) + "\n")
                        fout.flush()
                        buffer.clear()
            if buffer:
                fout.write("\n".join(json.dumps(x, ensure_ascii=False) for x in buffer) + "\n")
                fout.flush()


# =======================
# CLI
# =======================
def build_cfg(args) -> Dict[str, Any]:
    cfg = dict(CONFIG)
    if args.config_yaml:
        cfg = deep_merge(cfg, load_yaml(args.config_yaml))
    if args.llm_name:
        cfg["llm"]["name"] = args.llm_name
    if args.sem_num is not None:
        cfg["runner"]["sem_num"] = args.sem_num
    if args.batch_size is not None:
        cfg["runner"]["batch_size"] = args.batch_size
    if args.sampling_params_json:
        cfg["generation"]["sampling_params"] = json.loads(args.sampling_params_json)
        cfg["generation"]["beam_search"] = None
    if args.best_of_n is not None:
        cfg.setdefault("best_of_n", {})["enabled"] = bool(args.best_of_n)
    return cfg


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--config-yaml", default=None)
    parser.add_argument("--llm-name", default=None)
    parser.add_argument("--sem-num", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--sampling-params-json", default=None)
    parser.add_argument("--best-of-n", type=int, default=None, help="0/1")
    args = parser.parse_args()

    cfg = build_cfg(args)
    logging.basicConfig(level=getattr(logging, str(cfg.get("log_level", "INFO")).upper(), logging.INFO))
    if cfg["llm"]["name"] not in LLMFactory.reg:
        raise ValueError(f"Unknown llm-name={cfg['llm']['name']}. Available={list(LLMFactory.reg)}")
    await run(cfg, args.input_path, args.output_path)


if __name__ == "__main__":
    asyncio.run(main())
