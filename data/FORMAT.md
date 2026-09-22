# Long-Jev 统一基准数据格式（unified episode JSONL）

所有基准（3 游戏：maze/snake/pokemon；3 决策：webshop/alfworld/mind2web）统一为
同一种 episode JSONL，一行一个 episode，供三臂训练（C1 无记忆 / C3 线性 KV /
C4 两级记忆）与冒烟评测共用。

## Schema

```json
{"task": "pokemon", "episode_id": "pokemon-train-0001", "split": "train",
 "meta": {}, "success": true, "n_steps": 24,
 "steps": [
   {"t": 1, "obs": "...", "candidates": {"move:1": "...", "switch:2": "..."},
    "action": "move:1", "event": "used Psychic: dealt 41 percent; ..."}
 ]}
```

规则：
- `action` 必须是同一步 `candidates` 的某个 key；`candidates` 每步 3~8 个（上限 10），
  dict 顺序 = 呈现顺序，key 在步内稳定唯一
- `obs` ≤3500 字符（保证单条训练样本 prefix+最长候选叶子 <2048 token）
- `event` ≤200 字符，描述该动作导致的结果（供 C3/C4 记忆回放）
- 生成全程确定性（固定 seed）；train/dev/test 来源互不相交（官方划分或独立种子）
- 网格任务（maze*/snake*）沿用旧顶层键（size/seed/epsilon），step 结构一致

## 目录

| 基准 | 路径 | train/dev/test |
|---|---|---|
| maze（网格 5/8/16/32） | `data/maze{5,8,16,32}_{train,dev,test,ood}.jsonl`；smoke 视角 `data/benchmarks/maze/`（symlink→maze8） | BFS+ε 教师 |
| snake（网格 6/8/10/12） | `data/snake{6,8,10,12}_{train,dev,test}.jsonl`；smoke 视角 `data/benchmarks/snake/`（symlink→snake8） | BFS+flood-fill 教师 |
| pokemon | `data/benchmarks/pokemon/` | 自建 3v3 模拟器 + damage-race 启发式教师 |
| webshop | `data/benchmarks/webshop/`（raw/ 存官方数据） | 官方商品/查询子集 + 离线检索复刻，reward 按官方公式 |
| alfworld | `data/benchmarks/alfworld/`（raw/ 存官方轨迹） | 官方 json_2.1.1 专家轨迹 + 模板干扰候选 |
| mind2web | `data/benchmarks/mind2web/`（原始官方 JSON 同级） | train_10.json；test=官方 cross-domain 切分 |
| babilong（附录/迁移） | `data/benchmarks/babilong/` | 官方任务文件 |

每个基准目录内另有 `README.md`：数据来源与许可、教师/候选构造规则、再生成命令、统计。

## 消费方式

```bash
# 微量训练（C1 冒烟，原版 NanoJev warm-start）
python train_generic.py --task pokemon \
  --train ../../data/benchmarks/pokemon/train.jsonl \
  --dev   ../../data/benchmarks/pokemon/dev.jsonl \
  --checkpoint ../../checkpoints/NanoJev-unified --steps 60 --out ../../models/smoke_pokemon

# 步级准确率评测（零样本或训练后）
python eval_acc.py --task pokemon --test ../../data/benchmarks/pokemon/test.jsonl \
  --checkpoint ../../checkpoints/NanoJev-unified   # 或 ../../models/smoke_pokemon
```

C3/C4 记忆臂直接按 `steps` 顺序流式消费 obs/action/event（与网格任务同协议）。
