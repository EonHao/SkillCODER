<div align="center">

# SkillCODER

### 面向 Agent Skill 的语义水印与买家归因

[English](README.md) · [系统架构](docs/architecture.md) · [CLI 接口说明](docs/contracts.md) · [安全策略](SECURITY.md)

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-Apache--2.0-4C1)
![API](https://img.shields.io/badge/API-OpenAI--compatible-412991)
![Runtimes](https://img.shields.io/badge/Runtimes-Direct%20%7C%20LangChain%20%7C%20CAMEL-0B7285)

**论文已被 ACM CCS 2026（Round 2）接收**

</div>

SkillCODER 为 Markdown 格式的 Agent Skill 生成买家专属版本，并通过黑盒查询检测其中的语义水印。它把 Package 级语义解析、私钥控制的词语映射、模型在环改写、正负探针对照和纠错解码接入同一套命令行流程。

买家拿到的仍是一组可直接使用的 Markdown 文件。Owner key、码本、冻结探针与审计记录只保存在发行方。

## 工作方式

<p align="center">
  <img src="docs/assets/skillcoder-pipeline.png" alt="SkillCODER 论文主流程" width="100%">
</p>

<p align="center"><em>论文主流程包含水印嵌入、模型在环保真优化、黑盒差分探测和买家归因。</em></p>

构建过程先让模型读取整个 Skill Package，找出适合承载水印的语义位置。Owner key 随机决定 Buyer ID 与码字的对应关系，也控制词语方向、提示选择和载体位置。模型随后执行最多三轮生成、评审与修订，在保留原有行为的前提下，把受控词语分散写入合适的段落。

检测过程使用成对探针。Active 探针包含私钥选中的 cue，Decoy 探针使用格式相同的干扰 cue，Normal 查询负责测量日常任务中的误激活。系统根据三组响应的差异判断水印是否存在，再用 ECC 解码已发布的买家编号。

当前实现包括以下功能。

- 单文件和多文档 Skill Package
- Package 级 LLM 语义解析
- 私钥随机化的买家码本与词语映射
- 最多三轮模型在环生成、评审和修订
- 覆盖五类审计意图的正负探针
- Active、Decoy 与 Normal 三组行为统计
- ECC 解码与多买家归因
- Direct、LangChain 和 CAMEL 探测运行时
- 带完整性校验的审计记录与发布清单

## 安装

```bash
git clone https://github.com/EonHao/SkillCODER.git
cd SkillCODER
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

使用 LangChain 或 CAMEL 时再安装对应依赖。

```bash
pip install -e '.[langchain]'
pip install -e '.[camel]'
```

## 配置模型

SkillCODER 调用 OpenAI-compatible Chat Completions 接口。模型名称和 Base URL 均由用户配置，示例采用 OpenRouter 与 Qwen。

```bash
export SKILLCODER_MODEL_API_KEY='your-model-service-key'
export SKILLCODER_MODEL_BASE_URL='https://openrouter.ai/api/v1'
export SKILLCODER_MODEL='qwen/qwen3-max'
export SKILLCODER_OWNER_KEY="$(openssl rand -hex 32)"
```

Owner key 至少包含 32 个 UTF-8 字节，并且不能进入买家交付目录或公开日志。命令行参数 `--model` 和 `--base-url` 可以覆盖环境变量。

## 跑通一个示例

下面这条命令会依次生成查询、构建买家版本、执行探测、解码买家编号并写出报告。

```bash
skillcoder run \
  --source examples/code_review/SKILL.md \
  --skill-id code_review \
  --buyer-id buyer_1 \
  --buyer-count 8 \
  --codeword-length 4 \
  --normal-query-count 10 \
  --pairs 5 \
  --output run/code_review_buyer_1
```

运行完成后会生成以下文件。

```text
run/code_review_buyer_1/
├── normal_queries.json
├── report.json
├── release.json
└── package/
    ├── build.json
    ├── buyer_delivery/SKILL.md
    └── owner_audit/audit.json
```

`package/buyer_delivery/` 是候选交付内容。`report.json` 保存探测与解码结果，`release.json` 记录通过门禁后允许发布的文件。分发前可以再次验证清单与文件完整性。

```bash
skillcoder verify-release --run run/code_review_buyer_1
```

## 生成多个买家版本

`run-family` 先冻结一份共享语义计划，再生成多个买家版本。每个候选版本都会单独接受探测，满足门禁后才会写入认证发布清单。

```bash
skillcoder run-family \
  --source datasets/paper_skills/real_world/travel_planning/travel-planner \
  --entrypoint SKILL.md \
  --skill-id travel_planning \
  --buyer-count 8 \
  --codeword-length 4 \
  --normal-query-count 10 \
  --pairs 5 \
  --probe-runtime camel \
  --output run/travel_planning_family
```

默认发布门槛如下。

| 信号 | 门槛 |
|---|---|
| Active 激活率 | ≥ 60% |
| Decoy 激活率 | ≤ 20% |
| Normal 激活率 | ≤ 10% |
| 买家归因 | ECC 解码得到预期 Buyer |

## 检测可疑副本

检测时需要发行方保留的可信 release、待检查的 Skill Package 和一组 normal queries。

```bash
skillcoder probe-suspect \
  --reference run/travel_planning_family \
  --suspect evidence/suspected-skill \
  --entrypoint SKILL.md \
  --normal-queries run/travel_planning_family/normal_queries.json \
  --pairs 5 \
  --runtime langchain \
  --output evidence/detection-report.json
```

报告会给出水印检测分数、判定阈值、三组探针统计和 ECC 原始观测。解码成功时，报告还会返回与该副本匹配的已发布 Buyer。

正负探针共享同一个自然任务模板，只替换其中的 cue。模型负责生成模板，并检查任务相关性、意图匹配、语言自然度和 cue 的语义位置。探针覆盖策略检查、响应生成、下一步推理、升级判断与澄清请求。报告保留每一对探针的差分值，方便复核 Active 激活与 Decoy 抑制是否同时成立。

## 威胁模型

接收方可以查看和修改交付的 Skill，也可以了解公开算法。他可能执行同义改写、段落重排、内容压缩或多副本比较。发行方私有保存 Owner key、码本、审计记录和探针配置。检测方只能观察可疑 Agent 返回的文本。

系统关注仍保留主要任务能力的 Skill 变体。大幅删改可能同时破坏 Skill 功能和水印信号，这类退化会如实反映在探测报告中。构建与探测所用的模型端点会接触 Skill 内容，部署方应根据自己的数据要求选择服务商。

## 工程结构

```text
skillcoder/             核心实现
tests/                  接口、安全与对抗性测试
examples/code_review/   最小可运行示例
datasets/paper_skills/  固定版本的研究输入
docs/                   架构与接口文档
```

## 开发验证

```bash
pip install -e '.[test]'
pytest
python -m mypy --no-incremental skillcoder
python -m build
```

## 许可证

SkillCODER 源代码及项目原创材料采用 [Apache License 2.0](LICENSE)。

`datasets/paper_skills/real_world/` 收录的第三方 Skill 保留各自的上游许可证，不属于本项目的 Apache-2.0 授权范围。

- Trail of Bits `differential-review` 遵循 CC BY-SA 4.0，许可证全文保存在 `datasets/paper_skills/licenses/trailofbits-skills/LICENSE`
- `csv-data-summarizer` 的固定版本在上游 README 中声明 MIT，该版本没有包含 README 所链接的独立 `LICENSE` 文件
- ErlebnisW `travel-planner` 遵循 MIT，上游许可证全文随 Skill 保留

各项研究输入的来源仓库、commit、路径和许可证记录见 [`datasets/paper_skills/manifest.json`](datasets/paper_skills/manifest.json)。收录这些文件不会改变其著作权归属或许可证条款。

## 引用

```bibtex
@inproceedings{huang2026skillcoder,
  title     = {{SkillCODER}: Towards Auditing and Attribution of Copyright Infringement in {LLM} Agent Skills},
  author    = {Huang, Enhao and Xia, Chunshu and Li, Yiming and Yang, Yuchen and Yang, Bingrun and Qin, Zhan and Tao, Dacheng and Ren, Kui},
  booktitle = {Proceedings of the ACM SIGSAC Conference on Computer and Communications Security (CCS)},
  year      = {2026}
}
```
