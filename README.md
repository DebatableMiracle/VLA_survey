# VLA_survey
A supporting repository for the VLA survey around autonomous driving! 


# Datasets and benchmarks

| Benchmark | Year | Protocol | Scale | Main Metric | Main Limitation |
|---|---:|---|---|---|---|
| [CARLA](https://carla.org/) | 2017 | Reactive closed-loop | Synthetic | Driving score | Simulator domain gap |
| [BDD-X](https://github.com/JinkyuKimUCB/BDD-X) | 2018 | Language (explanation) | ~7K videos | Language metrics | Explanations written after the fact |
| [Talk2Car](https://github.com/talk2car/Talk2Car) | 2019 | Language (grounding) | Commands on nuScenes | Grounding accuracy | Single-object references |
| [nuScenes](https://www.nuscenes.org/) | 2020 | Open-loop | 1,000 scenes | L2, collision rate | Ego-status shortcut; mostly simple driving |
| [Waymo Open Dataset](https://waymo.com/open/) | 2020 | Open-loop | 1,150 segments | Task-specific | No language annotations |
| [nuPlan](https://www.nuscenes.org/nuplan) | 2021 | Closed-loop | >1,200 h, 4 cities | Closed-loop score | High compute cost |
| [DRAMA](https://usa.honda-ri.com/drama) | 2023 | Language (risk) | Risk localization and captions | Localization, captioning | Single important object |
| [NuScenes-QA](https://github.com/qiantianwen/NuScenes-QA) | 2024 | Language (VQA) | 34K scenes, 460K QA | Accuracy | Template-based questions |
| [Bench2Drive](https://github.com/Thinklab-SJTU/Bench2Drive) | 2024 | Reactive closed-loop | 220 routes, 44 scenarios | Driving score, success rate | Inherits the CARLA domain |
| [NAVSIM v1](https://github.com/autonomousvision/navsim) | 2024 | Non-reactive | 1,192 / 136 train/test scenarios | PDMS | Non-reactive; near saturation |
| [DriveLM-nuScenes](https://github.com/OpenDriveLab/DriveLM) | 2024 | Language (GVQA) | nuScenes keyframes | Accuracy, GPT score, match | Action imbalance; GPT-score validity |
| [NuInstruct](https://github.com/xmed-lab/nuinstruct) | 2024 | Language (instruction) | 91K QA, 17 subtasks | Task accuracy | nuScenes geography |
| [LingoQA](https://huggingface.co/papers/2312.14115) | 2024 | Language (video QA) | 28K scenarios, 419K annotations | Lingo-Judge | Learned judge can be gamed |
| [LaMPilot](https://github.com/PurdueDigitalTwin/LaMPilot) | 2024 | Language programs | Instruction-following tasks | Completion, collision | Simulated primitives |
| [AutoTrust](https://github.com/taco-group/AutoTrust) | 2024 | Trustworthiness | >10K scenes, 18K queries | Five dimensions | English only |
| [CODA-LM](https://coda-dataset.github.io/) | 2025 | Language (corner cases) | Road corner cases | LLM-judged scores | Judge-dependent |
| [DriveLMM-o1](https://github.com/ayesha-ishaq/DriveLMM-o1) | 2025 | Reasoning (QA) | Step-annotated QA | Step and answer scores | LLM-based step scoring |
| [NAVSIM v2](https://github.com/autonomousvision/navsim) | 2025 | Pseudo-simulation | navtest, navhard | EPDMS | Synthetic second-stage views |
 [DriveBench](https://github.com/worldbench/DriveBench) | 2025 | VLM evaluation across perception, prediction, behavior, and planning under clean, corrupted, and text-only inputs | 19,200 frames; 20,498 QA pairs; 17 settings | Accuracy, visual grounding, and reliability metrics | Reveals strong dependence on textual cues and general knowledge; benchmark performance can mask poor visual grounding |
| [SCD-Bench](https://arxiv.org/abs/2503.06497) | 2025 | Safety-cognition evaluation of VLMs in interactive driving scenarios | SCD-Bench; SCD-Training contains 324.35K samples | Safety cognition evaluation with LLM-based assessment | Focused specifically on safety cognition rather than general driving understanding |
| [nuScenesR²-6K](https://github.com/AMAP-ML/AutoDrive-R2) | 2026 | Reasoning (SFT) | 6,000 image-trajectory pairs | Trajectory error | Small scale |


# World Models and Roles
| Year | System | Space | Role |
|---|---|---|---|
| 2023 | [GAIA-1](https://arxiv.org/abs/2309.17080)† | Pixel | Video generation conditioned on video, text, and action |
| 2024 | [GenAD](https://github.com/OpenDriveLab/DriveAGI) | Pixel | Generalized video prediction for driving |
| 2024 | [Vista](https://github.com/OpenDriveLab/DriveAGI) | Pixel | Generalizable, action-controllable prediction |
| 2024 | [DriveDreamer](https://github.com/JeffWang987/DriveDreamer) | Pixel | Generation under traffic-structure conditions |
| 2024 | [DrivingWorld](https://github.com/YvanYin/DrivingWorld) | Pixel | Video-GPT-style generation |
| 2024 | [Drive-WM](https://github.com/BraveGroup/Drive-WM) | Pixel | Multi-view forecasting used for planning |
| 2024 | [OccWorld](https://github.com/Hualin1995/occWorld) | Occupancy | 4D occupancy forecasting |
| 2025 | [Epona](https://github.com/Kevin-thu/Epona) | Pixel | Autoregressive diffusion for long rollouts |
| 2025 | [DrivingGPT](https://openaccess.thecvf.com/content/ICCV2025/html/Chen_DrivingGPT_Unifying_Driving_World_Modeling_and_Planning_with_Multi-modal_Autoregressive_ICCV_2025_paper.html) | Pixel | Joint world modeling and planning |
| 2025 | [Drive-OccWorld](https://github.com/yuyang-cloud/Drive-OccWorld) | Occupancy | Occupancy forecasting coupled to planning |
| 2025 | [WoTE](https://github.com/liyingyanUCAS/WoTE) | BEV | Scores candidate trajectories online |
| 2025 | [LAW](https://github.com/BraveGroup/LAW) | Latent | Self-supervised auxiliary signal |
| 2025 | [SSR](https://github.com/PeidongLi/SSR) | Latent | Self-supervised auxiliary signal |
| 2025 | [World4Drive](https://github.com/ucaszyp/World4Drive) | Latent | Intention-aware trajectory selection |
| 2024 | [PIWM](https://github.com/gaoyinfeng/PIWM) | Latent | Models the intentions of other agents |
| 2025 | [Raw2Drive](https://github.com/Thinklab-SJTU/Raw2Drive) | Latent | Aligned world models for RL in CARLA |
| 2025 | [IRL-VLA](https://github.com/IRL-VLA/IRL-VLA)† | Latent | Reward world model for policy RL |
| 2026 | [DriveVLA-W0](https://github.com/BraveGroup/DriveVLA-W0)† | Pixel / tokens | Dense supervision that aids data scaling |
| 2026 | [DriveWorld-VLA](https://github.com/liulin815/DriveWorld-VLA)† | Latent | Latent world modeling inside a VLA |
| 2025 | [HERMES](https://github.com/lmd0311/hermes)† | BEV | Joint 3D understanding and generation |

† Systems marked with † also use language.

# References

