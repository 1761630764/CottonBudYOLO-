> 本项目致力于棉花顶芽检测与定位的模型算法开发、优化、应用，开发棉花顶芽检测与定位这个看似“小众”的场景里，针对模型算法优化升级到完整迭代闭环真正跑通，其意义远远不止“发一篇论文”或“跑个高分 mAP”。它本质上是在“高价值作物+密集种植+人工缺口”三大矛盾交汇点，撕开一个可复制、可扩张的“AI 精准农事”切口。

> - CottonBudYOLOv1为第1代版本，在田间得到到了实际验证证明其可行性，单行工作，工程应用性不灵活。
> - CottonBudYOLOv2为第2代版本，进一步进行优化，主要针对跨域检测性能进行优化，进一步兼容遮挡、小目标、复杂环境、单多行检测。加强其工程性，优化推理速度，优化CUDA加速，量化处理等等。
> - V1和V2版本均基于ultralytics库进行开发、优化和适配田间环境。

> CottonBudYOLOv1 项目地址：https://github.com/1761630764/CottonBudYOLOv1



**1、运行环境：**

```
conda create -n YOLOv11_UP python=3.10
Python==3.10  # 运行环境
pip install ultralytics==8.3.193  #  ultralytics指定版本安装8.3.193
pip show ultralytics # 查看ultralytics 版本号
```

**2、与v1相比更新内容：**

- [x] 111111
- [ ] 2222





