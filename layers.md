# Engine 分层

## 先理解这里的“层”

本文按 **Bazel target** 分层，而不是按目录分层。同一目录里的 target 可能属于不同层，例如：

- `//engine/app:loop` 属于 L0
- `//engine/app:targets` 属于 L1
- `//engine/app:app` 属于 L5

`A → B` 表示 A 依赖 B。L0-L6 是**职责带**，不是严格计算出来的依赖深度：依赖原则上由高层指向低层，同层内允许少量明确的单向依赖。

判断一个模块属于哪层，关键不是“它被谁依赖”，而是“它负责回答什么问题”。

## 一句话总览

先记住这一张表，后面都是展开：

| 层 | 一句话 | 比如 |
|---|---|---|
| L6 产品与集成 | 这个程序要**组合**哪些系统？ | `apps/world` 把 terrain + nav + sky + fog 装到一起 |
| L5 运行时宿主 | 怎样**启动并持续推进**一帧？ | window、swapchain、帧循环、input、clock |
| L4 帧渲染 | 一帧**画什么、pass 什么顺序**？ | renderer、IBL、post、rendergraph、volumetrics |
| L3 内容与离线组合 | 多个领域的数据怎样**导入、烘焙、打包**？ | glTF 导入、资源包、GI 烘焙、场景存档 |
| L2 专业子系统 | 怎样形成一项**完整、可验证**的专业能力？ | physics、geometry、UI、音频 system |
| L1 领域内核 | 一个独立领域的**数据模型和算法**是什么？ | mesh 数据、动画曲线、navmesh、纹理格式 |
| L0 基础边界 | 全工程共享的**最小词汇**是什么，OS/GPU 在哪里**隔离**？ | 数学、job system、RHI、platform |

记忆口诀：**L6 组合产品，L5 推进帧，L4 画一帧，L3 烘数据，L2 给能力，L1 定模型，L0 垫地基。**

## 分层图

```text
L6  apps/*
    产品、demo、端到端测试；跨子系统的组合根
    │
    ▼
L5  //engine/app:app
    window / device / swapchain / renderer / targets / input / clock
    │
    ▼
L4  //engine/render:render
    renderer / IBL / post / rendergraph / volumetric / decals /
    particles / fluid
    │
    ▼
L3  //engine/render:gi      //engine/asset:{asset,pack}
    //engine/serialize:serialize
    │
    ▼
L2  //engine/geometry:geometry    //engine/physics:physics
    //engine/text:ui              图片、纹理和音频适配器
    │
    ▼
L1  shaders / texture / anim / audio / nav / net / scene / ecs
    resource:stream / asset:texgen / app:targets
    │
    ▼
L0  core / resource / asset:json / app:loop
    rhi / platform / platform:{audio_out,gamepad}
```

## L6：产品与集成根

**核心问题：具体应用组合哪些系统，并如何逐帧驱动它们？**

`apps/*` 包含交互 demo、离屏集成测试和应用局部的 scene library。跨 renderer、physics、navigation、audio、assets、UI 的产品策略应留在这里。

“apps 是组合根”是架构约定，不是 Bazel visibility 强制规则；它也不表示每个 app 都必须依赖所有模块。

## L5：运行时宿主

**核心问题：怎样可靠地建立并推进窗口化帧循环？**

`//engine/app:app` 组装 window、device、swapchain、renderer、frame targets、input 和 clock，负责这些对象的生命周期，但不决定 gameplay 规则。

`engine/app` 不是一个整体层级：其中 `:loop` 和 `:targets` 是可以脱离 `:app` 使用的更低层 target。

## L4：帧渲染编排

**核心问题：一帧画什么，各个 pass 按什么顺序执行？**

`//engine/render:render` 消费 scene 和 geometry，管理 GPU 侧资源、材质、渲染路径与 RenderGraph，并通过 RHI 提交工作。

IBL、post、volumetric、decals、particles、fluid 和 rendergraph 当前都是 `:render` 的内部组成，不是独立 Bazel target。Render 层保持纯 C++，不直接暴露 Metal 类型。

## L3：内容与离线组合

**核心问题：怎样把多个领域的数据导入、转换、烘焙或持久化？**

- `//engine/asset:asset`：导入 glTF、图像和动画
- `//engine/asset:pack`：生成运行时友好的资源包
- `//engine/serialize:serialize`：保存 ECS/physics 的场景布置
- `//engine/render:gi`：借助 physics BVH，在 CPU 上烘焙辐照度体积

这些 target 会跨越多个领域，但产物仍是可复用数据；它们不拥有窗口或完整帧。

## L2：专业子系统与适配器

**核心问题：怎样把基础数据和算法组成一项完整、可验证的专业能力？**

代表 target 包括 geometry、physics、PNG/JPEG、纹理压缩、audio system、系统音频解码和 immediate-mode UI。

本层可以组合少量低层 target，但不编排完整世界或一帧。已知的同层单向依赖包括：

- `physics → geometry`：凸包和空间查询复用 geometry
- `jpeg → png`：复用图像数据表示

## L1：可复用领域内核

**核心问题：一个独立领域的稳定数据模型和算法是什么？**

包括 shaders、texture、anim、audio、nav、net、scene、ecs、`resource:stream`、`asset:texgen` 和 `app:targets`。

本层只依赖 L0，但并非“全部只依赖 core”：

- scene、ecs 还依赖 resource
- `app:targets` 依赖 rhi
- 大多数纯 CPU target 才是只依赖 core

## L0：基础与外部能力边界

**核心问题：全工程共享的最小词汇是什么，OS/GPU 能力在哪里终止？**

- `core`：数学和 job system；不依赖其他 engine target
- `resource`：供 scene/render 共享的稳定资源句柄
- `asset:json`、`app:loop`：无 engine 依赖的独立基础能力
- `rhi`：device、GPU resource 和 command submission；不认识 material、light、scene
- `platform`：窗口、字体、扬声器、手柄等 OS 能力

Objective-C++ 仅位于 `engine/rhi` 和 `engine/platform`。当前共有 5 个 `objc_library` target（rhi 1 个，platform 4 个：platform、audio_out、audio_file、gamepad）；上层 render/app 都是纯 C++ target。

## 最容易混的三组对比

### L1 vs L2 vs L3：模型、能力、组合

三层的区别在于**组合了几个领域**：

- **L1 只讲一个领域。** 问“这个数据结构长什么样”就能回答，不需要提别的领域。例如：动画曲线怎么插值（anim）、navmesh 的多边形长什么样（nav）、纹理像素怎么排（texture）。
- **L2 把 L1 做成能用的能力。** 问“给我一个能跑的 X 系统”，答案要拼几个 L1。例如：physics 要用 geometry 的凸包和 mesh；audio system 要把 clip 数据（audio）接到扬声器（platform:audio_out）上；UI 要把字体（platform）和 GPU 矩形（rhi/shaders）拼成可交互的控件。
- **L3 跨领域做数据，但不产生运行时行为。** 它的产物是文件、包、烘焙结果，运行时读进来直接用。例如：glTF 导入同时碰 mesh、动画、材质、图像四个领域；GI 烘焙同时碰 physics（BVH 打光线）和 render（辐照度体积格式）。

一句话区分：**L1 是名词（数据长什么样），L2 是动词（系统能干什么），L3 是工厂（离线把数据准备好）。**

常见误放：

- 把“场景怎么存盘”写进 scene（L1）→ 应该去 serialize（L3），因为存盘要同时理解 ECS、physics 和 json 三个领域。
- 把“光线打 BVH 烘辐照度”写进 physics（L2）→ 应该去 gi（L3），因为 physics 不应该知道辐照度体积是什么。
- 把“播一个音效”写进 audio clip（L1）→ 应该去 audio system（L2），因为出声需要 mixer + 输出设备，不只是一段波形数据。

### L4 vs L5：画一帧 vs 推一帧

- **L5 回答“帧从哪来”**：窗口还在不在、vsync 等多久、这一帧的 dt 是多少、input 有哪些事件。它不关心画面里有什么。
- **L4 回答“帧里有什么”**：scene 里有哪些物体、阴影 pass 在前还是在后、这个像素走前向还是延迟。它不关心窗口是谁创建的。

检验方法：把 L5 整个换掉（比如从窗口改成离屏 `apps/shot` 单帧输出），L4 的代码应该一行不改。如果改了，说明窗口逻辑漏进了渲染层。

### 为什么 L0 是四个东西，不是一个 core

传统引擎的“Core”文件夹常是个大杂烩，本仓库把它拆成了四个互不依赖的 L0 target，每个回答不同的边界问题：

| L0 成员 | 回答的问题 | 为什么必须在最底层 |
|---|---|---|
| `core`（math + jobs） | 全工程的数学语言和并行原语是什么？ | 所有层都要用它做计算，它不能反过来认识任何层 |
| `resource`（句柄） | 跨 scene/render 的资源身份是什么？ | scene（L1）和 render（L4）都要引用同一份资源，身份定义必须比它俩都低 |
| `app:loop`（clock + actions） | 时间步和输入的含义是什么？ | headless 测试也要推进时间，但不能拉起窗口；所以它必须脱离 `:app` 独立存在 |
| `rhi` / `platform` | GPU / OS 能力在哪里终止？ | 隔离 Apple 框架：Metal 类型和 AppKit 类型不许漏到上层，上层保持纯 C++ 可测 |

注意 `rhi` 连 `core` 都不依赖——一个 RHI 如果反过来要引用上层的数学类型，它就不是边界了。

## 附：3D 引擎的 core 一般解决哪些核心问题

上面是“本仓库的分层”，这里回答更一般的问题：**一个从零写的 3D 引擎，core 部分通常要解决什么？**

按“从纯计算到碰硬件”的顺序，一般有这七个：

1. **数学语言。** 向量、矩阵、四元数，以及全工程只定一次的约定：左手还是右手、上是 +Y 还是 +Z、矩阵行主序还是列主序、深度范围是 `[0,1]` 还是 `[-1,1]`、单位是米还是厘米。这些约定一旦分散到各模块各自为政，渲染、物理、动画三方的变换永远对不上。
2. **并行原语。** 蒙皮、剔除、建 BVH、光栅化 navmesh 体素——引擎里最热的循环都是“同一个独立运算跑几万次”。core 要给一个不会被误用出死锁的并行 for，而不是让每个模块自己开线程（两个模块各按机器核数开线程就会超订）。
3. **时间与输入的含义。** dt 怎么算、固定步长还是可变步长、一个按键动作是什么。这些定义必须脱离窗口和 GPU 可测试，否则 headless 机器跑不了逻辑测试。
4. **资源身份。** CPU 侧引用 GPU/磁盘资源时用的句柄是什么、生命周期谁管、streaming 时常驻/驱逐按什么规则。scene 和 renderer 都要说话，所以身份定义必须比它俩都低。
5. **数据交换基元。** JSON 解析、打包格式读写——无 engine 依赖、任何层都能用的读写能力。
6. **GPU 边界（RHI）。** 把 Metal/Vulkan/D3D 藏在一个只谈 buffer、texture、command 的接口后面：上层只说“画什么”，RHI 只管“怎么提交”。检验标准是 RHI 不认识 material、light、scene 这些词。
7. **OS 边界（platform）。** 窗口、字体、扬声器、手柄——每个都是独立的 OS 能力，互相不捆绑（开窗口的 target 不该因为链接了音频解码而变重）。

对照本仓库的位置：

| 一般 core 问题 | 本仓库在哪 | 层 |
|---|---|---|
| 数学语言 | `engine/core:core`（math.h 兼定右手/+Y/列主序/reversed-Z/米秒弧度） | L0 |
| 并行原语 | `engine/core:core`（jobs.h 的 ParallelFor，调用线程参与、无 future 防死锁） | L0 |
| 时间与输入含义 | `engine/app:loop`（clock、actions） | L0 |
| 资源身份 | `engine/resource` + `resource:stream`（常驻/驱逐策略） | L0 / L1 |
| 数据交换基元 | `engine/asset:json`、asset:pack | L0 / L3 |
| GPU 边界 | `engine/rhi`（唯一链 Metal、唯一能写 .mm 渲染代码的包） | L0 |
| OS 边界 | `engine/platform`（window/font/audio_out/audio_file/gamepad 各自独立） | L0 |

所以本仓库的 L0 不是“一个 core 文件夹”，而是**七个问题各自的最低承载点**：能只依赖 core 的就待在 `core` 里，必须碰 OS/GPU 的就钉死在 `rhi`/`platform` 里，时间语义和资源身份因为“被太多层共享”而同样沉到 L0。这也是判断新基础模块放哪的依据：先问它是上面七个中的哪一个，再问它最低能沉到哪。

## 关键依赖链

下面几条链比“目录从上到下”更能说明真实边界：

```text
app → render → gi → physics → geometry → shaders → core
app → targets → rhi
asset → jpeg → png → texture → core
serialize → physics / ecs / json / resource
text:ui → platform / rhi / shaders / core
audio:system → audio + platform:audio_out
```

## 新模块放哪一层

1. 先写出它唯一要回答的“核心问题”。
2. 如果它在组合多个领域，优先放 L3 以上；不要把编排逻辑塞进 L1/L2。
3. 如果它拥有一帧的 pass 顺序，属于 L4；如果它拥有窗口和帧循环，属于 L5。
4. 只有产品规则和端到端组合才进入 L6。
5. 检查依赖是否只指向同层的明确前置模块或更低层。
