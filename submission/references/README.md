# references

Public sources supporting this submission's design choices, per the
hackathon spec's requirement to justify structures, noise modeling and
the localization algorithm against credible sources.

## 1-8. Localization algorithm

Prior work supporting the reference-localization algorithm in
[`src/localizer/`](../src/localizer/).

The localizer is a Siamese ResNet-18 encoder → dense depthwise
cross-correlation → dilated-convolution context head → heatmap + sub-cell
offset decode. Each design decision below is annotated with the file that
implements it and the published result that motivates it.

---

## 1. Siamese matching by cross-correlation

The core formulation — encode template and search image with one shared
fully-convolutional network, then correlate the two feature maps to score
every candidate offset in a single pass.

| Paper | Relevance |
|---|---|
| Bertinetto et al., *Fully-Convolutional Siamese Networks for Object Tracking*, ECCV-W 2016 — [arXiv:1606.09549](https://arxiv.org/abs/1606.09549) | The SiamFC formulation this pipeline follows: a shared fully-convolutional embedding whose cross-correlation with a larger search region yields a dense score map in one forward pass. Establishes that the embedding must be translation-equivariant — no global pooling, no positional encoding — which is the hard constraint documented in [`encoder.py`](../src/localizer/encoder.py). |
| Li et al., *SiamRPN++: Evolution of Siamese Visual Tracking with Very Deep Networks*, CVPR 2019 — [arXiv:1812.11703](https://arxiv.org/abs/1812.11703) | Introduces **depthwise** cross-correlation: correlate channel-by-channel rather than collapsing to a scalar, cutting cost and memory while balancing the two branches for stable training. Also shows the resulting channels are near-orthogonal and semantically distinct. Directly supports the channel-wise design in [`correlation.py`](../src/localizer/correlation.py) and the decision to feed the full channel volume — not a scalar score — into the context head. |
| Cheng et al., *QATM: Quality-Aware Template Matching for Deep Learning*, CVPR 2019 — [arXiv:1903.07254](https://arxiv.org/abs/1903.07254) | Learned template matching as a differentiable layer, and the reason raw similarity is a weak signal: it explicitly models 1-to-1 vs. 1-to-many matching quality. Motivates using peak **margin** rather than absolute correlation score as confidence in [`decode.py`](../src/localizer/decode.py). |

## 2. Why a scalar correlation peak is not enough — repetitive-structure ambiguity

The failure mode this project exists to solve: a DRAM mat is a periodic
lattice, so a reference patch and its lattice translate are pixel-identical
and the correlation surface has genuinely tied maxima.

| Paper | Relevance |
|---|---|
| Doubek et al., *Image Matching and Retrieval by Repetitive Patterns*, ICPR 2010 — [PDF](https://cmp.felk.cvut.cz/~chum/papers/Doubek-ICPR10.pdf) | Formalizes shift ambiguity in lattice-structured scenes and represents repeating elements by shift-invariant descriptors. The same ambiguity DRAM word-line/bit-line arrays produce. |
| Fan, Wu & Hu, *Towards reliable matching of images containing repetitive patterns*, Pattern Recognition Letters 32(14) 2011 — [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167865511002509) · [PDF](https://nlpr.ia.ac.cn/2011papers/gjkw/gk2.pdf) | Shows local descriptors alone cannot resolve repetitive-pattern correspondence, and that matching must reason over a larger spatial region. This is the argument for the wide-receptive-field context head. |
| Yang et al., *SIFT Saliency Analysis for Matching Repetitive Structures*, Math. Problems in Engineering 2017 — [Wiley](https://www.hindawi.com/journals/mpe/2017/2878930/) | Confirms the same finding for classical feature matching and quantifies the degradation, supporting the ZNCC-baseline failure rate reported in [`README.md`](../README.md). |

## 3. Global receptive field via dilated convolutions

The context head needs a receptive field spanning the whole 226×226
correlation map, at full resolution, to see where the lattice terminates.

| Paper | Relevance |
|---|---|
| Yu & Koltun, *Multi-Scale Context Aggregation by Dilated Convolutions*, ICLR 2016 — [arXiv:1511.07122](https://arxiv.org/abs/1511.07122) | The context-module design this head is built on: exponentially spaced dilations expand the receptive field without losing resolution or coverage. The `(1,2,4,8,16,32,64)` schedule in [`context_head.py`](../src/localizer/context_head.py) is exactly this construction. |
| Chen et al., *DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs*, TPAMI 2018 — [arXiv:1606.00915](https://arxiv.org/abs/1606.00915) | The stride-removal-plus-dilation trick used verbatim in [`encoder.py`](../src/localizer/encoder.py): set a late stage's stride to 1 and dilate its convolutions to keep the receptive field while halving output stride. Establishes output stride as the resolution/accuracy knob for dense prediction. |
| Wang et al., *Understanding Convolution for Semantic Segmentation* (HDC), WACV 2018 — [arXiv:1702.08502](https://arxiv.org/abs/1702.08502) | Identifies the **gridding artifact** in pure exponential dilation stacks — nearby cells sample disjoint grids — and fixes it by varying dilation rates so they share no common factor. Motivates the trailing dilation-1 layers appended to the schedule in [`context_head.py`](../src/localizer/context_head.py). |
| Yu et al., *Dilated Residual Networks*, CVPR 2017 — [arXiv:1705.09914](https://arxiv.org/abs/1705.09914) | Degridding by adding low-dilation layers at the end of a dilated stack, and the residual formulation used in `_DilatedBlock`. |

## 4. Padding, borders, and translation equivariance

The head uses `padding_mode="replicate"` rather than PyTorch's zero-padding
default. This is not a detail — with zero padding, training collapsed to a
fixed corner regardless of input.

| Paper | Relevance |
|---|---|
| Islam et al., *How Much Position Information Do Convolutional Neural Networks Encode?*, ICLR 2020 — [arXiv:2001.08248](https://arxiv.org/abs/2001.08248) | The direct explanation: **zero-padding is the mechanism by which CNNs encode absolute position**. A model that must be translation-equivariant therefore cannot use it. This is the published result behind the replicate-padding decision documented in [`context_head.py`](../src/localizer/context_head.py). |
| Kayhan & van Gemert, *On Translation Invariance in CNNs: Convolutional Layers can Exploit Absolute Spatial Location*, CVPR 2020 — [arXiv:2003.07064](https://arxiv.org/abs/2003.07064) | Shows filters learn to fire at fixed absolute locations by exploiting image-boundary effects, and that with a large receptive field this leaks *far from the border* — precisely the regime of a dilation-64 layer. Their fix is removing the boundary-induced location cue, which is what replicate padding does here. |
| Islam, Kowal, Jia, Derpanis & Bruce, *Position, Padding and Predictions: A Deeper Look at Position Information in CNNs* — [arXiv:2101.12322](https://arxiv.org/abs/2101.12322) | Follow-up analysis of padding-induced border artifacts and their effect on predictions near image edges; covers alternative padding modes. |
| Zhang, *Making Convolutional Networks Shift-Invariant Again*, ICML 2019 — [arXiv:1904.11486](https://arxiv.org/abs/1904.11486) | Strided downsampling violates the sampling theorem and destroys shift-invariance. Supports keeping output stride at 4 rather than 8 in the encoder, and using no pooling at all in the context head. |

## 5. Heatmap targets, sub-pixel offsets, and the loss

The head predicts a heatmap plus a sub-cell offset field, trained with a
penalty-reduced focal loss against a soft Gaussian target.

| Paper | Relevance |
|---|---|
| Law & Deng, *CornerNet: Detecting Objects as Paired Keypoints*, ECCV 2018 — [arXiv:1808.01244](https://arxiv.org/abs/1808.01244) | Origin of the **penalty-reduced focal loss** and the unnormalized 2D Gaussian target implemented in [`losses.py`](../src/localizer/losses.py) and [`targets.py`](../src/localizer/targets.py). Their ablation shows the Gaussian penalty reduction is worth 5–6 AP over hard labels — the empirical case against hard 0/1 cell labels. |
| Zhou et al., *Objects as Points* (CenterNet), 2019 — [arXiv:1904.07850](https://arxiv.org/abs/1904.07850) | The keypoint-heatmap + local-offset-regression decode this model uses: predict a peak, regress the sub-cell residual to recover resolution lost to output stride, extract local maxima instead of running full NMS. Matches [`decode.py`](../src/localizer/decode.py) and the offset head. |
| Lin et al., *Focal Loss for Dense Object Detection*, ICCV 2017 — [arXiv:1708.02002](https://arxiv.org/abs/1708.02002) | The extreme foreground/background imbalance argument. Here it is one positive cell against 51,075 negatives — a more severe ratio than dense detection — which is why the focal formulation, and the `-4.0` heatmap-head bias initialization, are load-bearing. |
| Shrivastava et al., *Training Region-based Object Detectors with Online Hard Example Mining*, CVPR 2016 — [arXiv:1604.03540](https://arxiv.org/abs/1604.03540) | Justifies the explicit `hard_negative_loss` margin term in [`losses.py`](../src/localizer/losses.py): the informative negatives are a small, specific set (the periodic twins), and sampling by current loss beats uniform treatment. |

## 6. Backbone and normalization

| Paper | Relevance |
|---|---|
| He et al., *Deep Residual Learning for Image Recognition*, CVPR 2016 — [arXiv:1512.03385](https://arxiv.org/abs/1512.03385) | The ResNet-18 backbone. Only stem + layer1 + layer2 are used — the task needs spatial resolution, not the class-level abstraction layer3/layer4 build. |
| Wu & He, *Group Normalization*, ECCV 2018 — [arXiv:1803.08494](https://arxiv.org/abs/1803.08494) | Why `nn.GroupNorm` and not BatchNorm in the context head: BN's error rises sharply at small batch size, and 1000×1000 search images force small batches. GN's statistics are batch-independent. |

## 7. Synthetic training data and sim-to-real

The entire training set is procedurally generated — no proprietary geometry,
no real fab imagery.

| Paper | Relevance |
|---|---|
| Tobin et al., *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, IROS 2017 — [arXiv:1703.06907](https://arxiv.org/abs/1703.06907) | The core justification for the randomized imaging-noise pipeline: with enough simulator variability, the real domain becomes just another variation. Supports randomizing shot noise, astigmatism, drift, charging, and vignetting rather than trying to match one true noise model. |
| *Addressing Class Imbalance and Data Limitations in Advanced Node Semiconductor Defect Inspection: A Generative Approach for SEM Images*, 2024 — [arXiv:2407.10348](https://arxiv.org/abs/2407.10348) | Domain-specific precedent for synthesizing SEM imagery — including replicating SEM noise characteristics and surface roughness — to overcome data scarcity in semiconductor inspection. |
| *Industrial wafer edge segmentation for alignment: a dual-resolution deep learning approach with synthetic pretraining and unsupervised geometry regularization*, J. Intelligent Manufacturing 2026 — [Springer](https://link.springer.com/article/10.1007/s10845-026-02912-5) | Closest published analogue to this problem: deep learning for **wafer alignment**, trained with synthetic pretraining, using multi-resolution features. Independent confirmation that synthetic-pretrain-then-adapt is the accepted approach for this task class. |

## 8. Fine-tuning without forgetting

Relevant to the mixed-replay fine-tune used to adapt the model onto
AFB-rendered data (`scripts/finetune_on_manifest.py`) without losing
harsh-noise robustness.

| Paper | Relevance |
|---|---|
| Kirkpatrick et al., *Overcoming catastrophic forgetting in neural networks*, PNAS 2017 — [arXiv:1612.00796](https://arxiv.org/abs/1612.00796) | The canonical statement of catastrophic forgetting during sequential training, and the stability–plasticity tradeoff. Frames the observed regression on harsh-noise conditions after domain fine-tuning, and motivates mixing a replay buffer of the original distribution into the fine-tune set. |

## 9. Semiconductor structures and SEM imaging noise

Public sources backing this submission's **synthetic-data and
noise-modeling** design choices (the generator in
[`src/pipeline.py`](../src/pipeline.py),
[`src/sem_imaging.py`](../src/sem_imaging.py) and
[`src/patterns/`](../src/patterns/)), per the hackathon spec's requirement
to justify structures and augmentations against credible sources.

**DRAM 1T-1C cell structure** (word lines, bit lines, capacitor storage)
- imec, [DRAM peripheral transistors technology platform](https://www.imec-int.com/en/articles/technology-platform-thermally-stable-dram-peripheral-transistors)
- SemiAnalysis, [The Memory Wall: Past, Present, and Future of DRAM](https://newsletter.semianalysis.com/p/the-memory-wall)

**SEM imaging noise and degradation modeling**
- [Correction of Scanning Electron Microscope Imaging Artifacts in a Novel Digital Image Correlation Framework](https://pmc.ncbi.nlm.nih.gov/articles/PMC6541586/), *Experimental Mechanics* (Springer)
- [Scanning Electron Microscope Image Signal-to-Noise Ratio Monitoring for Micro-Nanomanipulation](https://hal.science/hal-01051309/document)

**Data augmentation for scale/rotation robustness in matching tasks**
- [An Efficient Deep Template Matching and In-Plane Pose Estimation Method via Template-Aware Dynamic Convolution](https://arxiv.org/html/2510.01678), arXiv
- [Who Handles Orientation? Investigating Invariance in Feature Matching](https://arxiv.org/html/2604.11809v1), arXiv
