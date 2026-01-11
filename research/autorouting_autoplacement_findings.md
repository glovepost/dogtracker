# KiCad Autorouting and Autoplacement Research Findings

**Last Updated:** January 2026

## 1. Executive Summary

This document summarizes findings on automating PCB layout in KiCad 8/9, focusing on autorouting and component placement. The goal is to identify improvements over the current heuristic-based placement and Java-subprocess-based Freerouting integration. Additional cutting-edge research as of January 2026 highlights significant advancements in AI-driven tools, GPU acceleration, and integrated workflows.

### Key Takeaways:
*   **Autorouting:** Freerouting remains viable with its Python client, but new tools like OrthoRoute (GPU-accelerated) and Autocuro (AI-based) offer faster, more efficient routing for KiCad 9+. DeepPCB has evolved with Pro features, though results vary. Emerging AI routers like Flux AI and custom grid-based solutions demonstrate rapid progress in handling complex designs.
*   **Autoplacement:** Heuristics in `smart_placer.py` align with best practices, but AI tools like Quilter (physics-driven RL), DeepPCB's placement optimization, Zuken AIPR, and Cadence Allegro X AI enable automated, constraint-aware placements that reduce design time significantly.
*   **KiCad 9 Migration:** IPC API enables advanced plugins; migration is essential as SWIG is removed in KiCad 10. New AI integrations leverage this for end-to-end automation.

---

## 2. Autorouting

### 2.1. Freerouting (Current Standard)

**Status:** Active, v2.1.0 with official Python client

**Current Implementation (`module_autorouter.py`):**
*   Java JAR file via `subprocess`, DSN export → routing → SES import
*   Module-specific settings (RF mode, trace widths, max passes)
*   Works but fragile (Java dependency, file I/O overhead)

**New: Official Python Client Library**

```python
# Installation
pip install freerouting-client

# Usage
from freerouting import FreeroutingClient
import os

api_key = os.environ.get("FREEROUTING_API_KEY")
client = FreeroutingClient(api_key=api_key)

# Check API status
status = client.get_system_status()

# Complete routing workflow
result = client.run_routing_job(
    dsn_file="module.dsn",
    ses_file="module.ses",
    router_settings={"max_passes": 100},
    polling_interval=5,
    timeout=300
)
```

**API Capabilities:**
*   `create_session()` - Start routing session
*   `run_routing_job()` - Complete workflow (upload, route, download)
*   Status polling with progress feedback
*   Cloud-based routing via `api.freerouting.app`

**Pros:**
*   No Java subprocess management
*   Better error feedback and progress monitoring
*   Cloud option eliminates local resource constraints

**Cons:**
*   Requires API key (request at freerouting.app)
*   Cloud dependency for full features
*   Still in alpha

**Recommendation:** Hybrid approach - keep local JAR for offline/fast routing, add Python client as optional cloud backend for complex boards.

**Sources:**
*   [Freerouting Python Client GitHub](https://github.com/freerouting/freerouting-python-client)
*   [Freerouting Documentation](https://freerouting.org/)

---

### 2.2. OrthoRoute (GPU-Accelerated, KiCad 9+)

**Status:** Production-ready for KiCad 9, updated in late 2025 with improved compatibility.

**What It Is:** GPU-accelerated autorouter using NVIDIA CUDA via CuPy, implementing Lee's algorithm (wavefront propagation) with massive parallelization. Recent updates include better handling of high-density designs and integration with KiCad's IPC API.

**Performance:**
*   512 net board: ~2 minutes
*   8,192 net backplane (200x160mm): routed successfully
*   Complex board: 44,233 vias, 68,975 tracks in 41 hours (vs. months for FreeRouting)
*   2025 benchmarks show 20-50% faster routing on latest NVIDIA GPUs.

**Technical Details:**
*   CUDA SSSP (parallel Dijkstra) for pathfinding
*   Nets routed sequentially on shared congestion map
*   Parallelism inside shortest-path search kernel
*   Requires significant VRAM for large boards (33.5GB for extreme case)
*   New features: Manhattan lattice enhancements for reduced via count.

**Requirements:**
*   KiCad 9.0+ (uses new IPC API plugin system)
*   NVIDIA GPU with CUDA support (falls back to CPU)
*   CuPy Python library

**Recommendation:** Ideal for complex, high-net-count boards; integrate via IPC API for our pipeline. Test on GPU-equipped systems for performance gains.

**Sources:**
*   [OrthoRoute Page](https://bbenchoff.github.io/pages/OrthoRoute.html)
*   [OrthoRoute GitHub](https://github.com/bbenchoff/OrthoRoute)
*   [KiCad Forum Discussion](https://forum.kicad.info/t/orthoroute-or-a-new-autorouter-plugin-for-ipc-apis/65177)
*   [Hackster.io Review](https://www.hackster.io/news/brian-benchoff-s-orthoroute-gives-kicad-gpu-accelerated-autorouting-powers-for-hefty-designs-fa5e414c738a)

### 2.3. DeepPCB (AI Cloud Router)

**Status:** Active, DeepPCB Pro launched September 2024, with 2025 updates including enhanced AI placement integration.

**Technology:**
*   Reinforcement learning (RL) trained through millions of simulations
*   Cloud-native on Google Cloud infrastructure
*   AI-driven placement feature (new), optimizing for space and routing challenges

**Claimed Benefits:**
*   DRC-clean layouts, minimizes vias
*   One manufacturer reported 50%+ design time reduction
*   2025 Pro updates: Human-like placements, faster iterations.

**Real-World Results (Mixed):**
*   EEVblog testing: underwhelming vs. traditional routers
*   AI doesn't understand circuit priorities
*   RF traces and strategic placement require manual intervention
*   Results often look "un-designed" to experienced engineers

**Pricing:** Free trial, pay-as-you-go AI credits

**Recommendation:** Low priority - our modules are simple enough for Freerouting, privacy concerns with uploading designs. Consider for benchmark testing.

**Sources:**
*   [DeepPCB Official](https://deeppcb.ai/)
*   [InstaDeep DeepPCB Pro Announcement](https://instadeep.com/2024/09/instadeep-introduces-deeppcb-pro-an-ai-powered-pcb-design-tool/)
*   [EEVblog Review](https://www.eevblog.com/forum/blog/eevblog-1535-deeppcb-ai-autorouting-tested!/)

### 2.4. Autocuro (AI Automation for KiCad)

**Status:** Released in 2025, supports KiCad 8/9 with end-to-end automation.

**What It Is:** AI-driven tool for schematic-to-layout automation, including placement, routing, power planes, and vias. Processes designs in ~5 minutes.

**Technical Details:**
*   Reads schematic intent before layout generation
*   Handles STM32-based boards effectively in tests
*   Minimizes vias, ensures DRC compliance

**Pros:**
*   Offline-capable after initial setup
*   Integrates with KiCad files
*   Superior to traditional autorouters for simple to medium complexity

**Cons:**
*   May require manual tweaks for critical paths
*   Still maturing for ultra-complex RF designs

**Recommendation:** Test integration for our modules; could replace manual routing steps.

**Sources:**
*   [Autocuro Blog: Automating KiCad Routing](https://autocuro.com/blog/how-we-automate-kicad-pcb-routing)

### 2.5. Flux AI Auto-Layout

**Status:** Updated Winter 2025 with cleaner routing and human-like patterns.

**What It Is:** Browser-based AI for PCB layout, focusing on clean traces, fewer vias, and manufacturable designs.

**Technical Details:**
*   Single-click routing with pattern recognition
*   Improvements: Shorter traces, placement-aware paths
*   Best for 80% automation, manual for constraints

**Pros:**
*   Fast results, collaborative workflow
*   Integrates with Flux's eCAD platform

**Cons:**
*   Cloud-dependent, potential privacy issues

**Recommendation:** Explore for rapid prototyping; align with our offline needs.

**Sources:**
*   [Flux AI Winter Update](https://www.flux.ai/p/blog/ai-auto-layout-winter-update)

### 2.6. Custom Grid-Based Autorouters

**Status:** Experimental, e.g., Peter Schmidt-Nielsen's 2025 implementation for memory interfaces.

**What It Is:** Reads KiCad PCB files, uses grid-based routing on multiple layers, outputs traces.

**Performance:**
*   Routed complex interface in 730ms with 5 layers
*   Outperforms Freerouting/DeepPCB on specific cases

**Recommendation:** Inspire custom enhancements to our pipeline for high-speed interfaces.

**Sources:**
*   [X Post by Peter Schmidt-Nielsen](https://x.com/ptrschmdtnlsn/status/1936535704911372548)

---

## 3. Autoplacement

### 3.1. Current Implementation Analysis (`smart_placer.py`)

| Feature | Implementation | Status |
|---------|---------------|--------|
| Topology detection | IC-centric, RF linear, power chain, bridge | Working |
| Component classification | Decoupling, RF, passives | Working |
| Netlist-aware placement | Connection-weighted positioning | Working |
| Collision detection | Bounding box overlap checking | Working |
| Force-directed refinement | Basic attraction/damping loop | **Partial** |

**Verdict:** Our topology-aware, netlist-weighted approach is well-aligned with academic best practices. The "IC-centric with decoupling proximity rules" is exactly what research recommends.

**Gap:** Force-directed refinement code (`_optimize_placement_iteration`) exists but isn't called in the main workflow.

---

### 3.2. Force-Directed Placement (Academic Background)

**Foundational Work:** Quinn & Breuer (1979) established force-directed PCB placement:
*   Phase I: Solve equations based on interconnection topology for optimal relative locations
*   Phase II: Resolve component overlaps (slot assignment)

**Algorithm Classifications:**
1.  **Classical:** Accumulated force models, energy minimization, combinatorial optimization
2.  **Hybrid:** Parallel/hardware accelerated, multilevel, multidimensional scaling

**Sources:**
*   [Force-directed Survey (arXiv)](https://arxiv.org/abs/2204.01006)
*   [Quinn & Breuer IEEE Paper](https://ieeexplore.ieee.org/document/1084652/)
*   [Modern PCB Placement DAC 2024](https://dl.acm.org/doi/10.1145/3649329.3663495)

---

### 3.3. Simulated Annealing for PCB Placement

**Available Tools:**

1.  **SA-PCB (OpenROAD Project)** - C++ annealer for polygonal components, 90°/45°/free rotation, BEN-AMEUR cost normalization, Timberwolf cooling schedule
    *   Repository: `github.com/The-OpenROAD-Project/SA-PCB`

2.  **simanneal (Python)** - General-purpose SA library, easy integration
    *   Repository: `github.com/perrygeo/simanneal`

**Integration Example:**
```python
from simanneal import Annealer

class PCBPlacer(Annealer):
    def move(self):
        comp = random.choice(self.components)
        comp.x += random.uniform(-1, 1)
        comp.y += random.uniform(-1, 1)

    def energy(self):
        return self._wire_length() + self._overlap_penalty() * 1000

placer = PCBPlacer(components, netlist, bounds)
placer.steps = 50000
state, energy = placer.anneal()
```

### 3.4. Quilter (Physics-Driven AI Placement)

**Status:** 2025 updates emphasize multi-candidate generation and native KiCad support.

**What It Is:** AI tool generating complete layouts from KiCad files using reinforcement learning and physics simulations.

**Technical Details:**
*   Uploads projects, defines constraints
*   Produces multiple layout candidates in parallel
*   Runs physics-aware checks (SI/PI/thermal)
*   Reduces TCO by 30-40% via fewer manual hours

**Pros:**
*   Native file output for polish
*   Handles placement and routing holistically

**Cons:**
*   Requires cloud access
*   May require verification for edge cases

**Recommendation:** Integrate as an add-on to `smart_placer.py` for complex optimizations.

**Sources:**
*   [Quilter Blog: Efficient PCB Layout Software](https://quilter.ai/blog/a-review-of-the-most-efficient-pcb-layout-software-in-2025-how-quilter-stacks-up)
*   [Quilter TCO Analysis](https://www.quilter.ai/blog/the-true-cost-of-enterprise-pcb-tools-in-2025-a-tco-analysis-of-altium-cadence-and-ai-powered-platforms)

### 3.5. Zuken CR-8000 AIPR (Autonomous Intelligent Place and Route)

**Status:** Mature in 2025, leverages machine learning for efficiency.

**What It Is:** AI for autonomous placement and routing, trained on design examples.

**Technical Details:**
*   3-stage approach: Constraint-driven, optimization, verification
*   Reduces design time by up to 30%
*   Applicable to broad user base via vast library

**Pros:**
*   High accuracy for complex boards
*   Integrates with CR-8000 platform

**Cons:**
*   Not native to KiCad; requires adaptation

**Recommendation:** Study for inspiration in enhancing our heuristics.

**Sources:**
*   [Zuken Blog: Future of AI PCB Design](https://www.zuken.com/us/blog/exploring-the-future-of-ai-based-pcb-design-solutions/)
*   [Medium Article: AI in PCB Design](https://medium.com/%40aa.suryanegara2/ai-in-pcb-design-transforming-electronics-manufacturing-6f16837a5167)

### 3.6. Cadence Allegro X AI

**Status:** 2025 enhancements for small-to-medium designs.

**What It Is:** Automates placement and routing, reducing tasks from days to minutes.

**Technical Details:**
*   Generative AI for layout options
*   Optimizes trace routing, equivalent quality to manual

**Pros:**
*   Transformative time savings
*   Built-in analysis

**Cons:**
*   Platform-specific; port ideas to KiCad

**Recommendation:** Benchmark against our tools for performance insights.

**Sources:**
*   [EMA-EDA Blog: Best AI for Circuit Design](https://www.ema-eda.com/ema-resources/blog/best-ai-for-circuit-design-and-analysis-in-2025-emd/)

---

## 4. KiCad 9 IPC API

### 4.1. Overview

**Breaking Change:** SWIG-based Python bindings deprecated in KiCad 9, removed in KiCad 10.

**New Architecture:**
*   Stable, language-agnostic interface via Protocol Buffers
*   Communication via Unix domain sockets (macOS/Linux) or named pipes (Windows)
*   External processes connect to running KiCad instance
*   2025 updates: Enhanced plugin support for AI tools like OrthoRoute and Autocuro.

| Aspect | SWIG (KiCad 8) | IPC API (KiCad 9+) |
|--------|----------------|-------------------|
| Stability | Breaks with refactoring | Stable interface |
| Language | Python only | Any language |
| Scope | Direct object access | Message-based |
| Status | **Deprecated** | Official |

**Migration Path:**
```python
# Current (SWIG)
import pcbnew
board = pcbnew.LoadBoard(path)
fp.SetPosition(pcbnew.VECTOR2I(...))

# Future (IPC API via kicad-python)
from kicad import KiCadAPI
api = KiCadAPI()
board = api.open_board(path)
api.set_footprint_position(ref, x, y)
```

**Sources:**
*   [KiCad IPC API Docs](https://dev-docs.kicad.org/en/apis-and-binding/ipc-api/)
*   [kicad-python PyPI](https://pypi.org/project/kicad-python/)
*   [KiCad Forum Discussion](https://forum.kicad.info/t/kicad-9-0-python-api-ipc-api/57236)

---

## 5. Recommendations for Next Steps

### 5.1. Short-Term (Current Architecture)

1.  **Complete Force-Directed Refinement:**
    *   Call `_optimize_placement_iteration` in main placement workflow
    *   Run 30-50 iterations after initial topology-based placement
    *   Check convergence before applying final positions

2.  **Add Pre-Routing Validation:**
    ```python
    def validate_for_routing(board):
        issues = []
        for fp in board.GetFootprints():
            for pad in fp.Pads():
                if not pad.GetNet():
                    issues.append(f"Unconnected: {fp.GetReference()}.{pad.GetNumber()}")
        return issues
    ```

3.  **Optional Freerouting Python Client:**
    *   Add as fallback/alternative to local JAR
    *   Better progress monitoring and error feedback

4.  **Integrate OrthoRoute:** Test GPU acceleration for high-net designs via IPC API.

5.  **Test Autocuro/Flux:** Evaluate for automated routing in simple modules.

### 5.2. Medium-Term (KiCad 9 Migration)

1.  **Create API Abstraction Layer:**
    ```python
    class KiCadBoard:
        @classmethod
        def load(cls, path):
            return IpcApiBoard(path) if KICAD_VERSION >= 9 else SwigBoard(path)
    ```

2.  **Evaluate OrthoRoute** for complex routing scenarios when GPU available

3.  **Adopt Quilter/Zuken-Inspired AI:** Enhance `smart_placer.py` with RL-based candidate generation.

### 5.3. Low Priority

1.  **DeepPCB Testing:** Manual test with one complex board for comparison
2.  **Simulated Annealing:** Optional refinement step using `simanneal`
3.  **Custom Grid Router:** Develop inspired by Schmidt-Nielsen for specific interfaces.

---

## 6. Implementation Priority

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 1 | Complete force-directed refinement in smart_placer.py | Low | Medium |
| 2 | Add pre-routing validation | Low | High |
| 3 | Optional Freerouting Python client | Medium | Medium |
| 4 | Integrate OrthoRoute via IPC API | Medium | High |
| 5 | Test Autocuro/Flux for automation | Low | Medium |
| 6 | KiCad 9 IPC API abstraction layer | High | Future-proofing |
| 7 | Quilter AI evaluation | Medium | High |
| 8 | Simulated annealing placement | Medium | Low |
| 9 | DeepPCB evaluation | Low | Research |

---

## 7. Conclusion

Our current implementation is well-architected and follows industry best practices. The main opportunities are:

1.  **Completing existing features** - Force-directed refinement code exists but isn't fully utilized
2.  **Adding robustness** - Pre-routing validation prevents common failure modes
3.  **Future-proofing** - Plan for KiCad 9 IPC API migration
4.  **Incorporating Cutting-Edge AI/GPU Tools** - OrthoRoute, Quilter, and Autocuro represent 2025 advancements that could automate 80% of our workflows, with manual oversight for critical aspects.

The heuristic, topology-aware placement is superior to generic AI/ML for our use case. Deterministic rules like "decoupling cap 1.5mm from IC VDD pin" are more reliable than learned approximations. However, hybrid AI-heuristic approaches show promise for efficiency gains.

**Bottom Line:** Invest in refining what we have rather than chasing new tools. The biggest wins come from completing the force-directed optimization loop, adding validation, and selectively integrating GPU/AI enhancements.
