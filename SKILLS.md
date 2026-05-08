# Technical Knowledge Requirements for Nemorix

> Map of what you know, what you need to learn, and realistic timelines.

---

## What You Already Have ✓

Based on the code and tests you've written:

- ✅ **Python 3.10+** — dataclasses, type hints, protocol classes, mock testing
- ✅ **Core CS fundamentals** — memory hierarchies, eviction algorithms, scheduling, discrete-event simulation
- ✅ **Testing discipline** — unit tests, integration tests, edge-case validation, determinism verification
- ✅ **Physics reasoning** — latency modeling, bandwidth calculations, cost formulas, scale analysis
- ✅ **Documentation** — clear README, cross-checks with hardware specs, inline comments
- ✅ **Git basics** — project structure, clean commits (implied)
- ✅ **Linux / Windows development** — working in terminals (PowerShell here), package management (pip), PYTHONPATH
- ✅ **Algorithm design** — weighted eviction scoring, competing objectives (recency + importance + priority + recompute cost)

You are **not** starting from zero. You can build the next phases.

---

## Path 1: Writing the Peer-Reviewed Paper

### Knowledge you need but don't have yet

| Knowledge area | Why needed | Effort to learn | Timeline |
|---|---|---|---|
| **LaTeX for academic writing** | Most physics/ML papers use LaTeX for typesetting (ACM, IEEE templates) | 2–3 days to basics, 1 week to proficiency | Week 1 |
| **Academic writing style** | Papers have specific structure (lit review, technical depth, formal claims) vs. simulation write-ups | Read 5 OSDI/ATC papers carefully, then write one section | Week 1 |
| **Experimental methodology** | How to design a fair hardware experiment with baselines, controls, and error bars | Read MLPerf Inference paper for benchmarking best practices | Week 2 |
| **Competitive landscape** | Know vLLM, Dynamo, LMCache, FlexGen deeply enough to claim novelty correctly | Read each paper's architecture section + try running them | Week 2–3 |
| **Related work section** | Cite OS paging theory, OS scheduling, cache replacement, ML inference, GPU memory management | Skim 20–30 papers, cite them correctly | Week 3 |
| **How peer review works** | Reviewers will nitpick; you must anticipate and defend | Read 3 accepted papers from your target venue | Week 1 |
| **How to respond to reviews** | Your paper will get rejected or need revision; responding well is a skill | Read examples of strong author responses | Month 3–4 |

### Concrete learning path (let's say you target MLSys 2026, deadline Sep 2025)

**Week 1: LaTeX + writing style**
- Install **Overleaf** (free, web-based LaTeX editor) and open an ICML/MLSys template
- Read the ICML style guide: 50 page limit including everything, specific font sizes, citation format
- Copy the template, rename sections to match your paper outline
- Write **10 pages** of sections 1–2 (intro, background) just to get in the flow

**Week 2: Experimental design**
- Read: "Benchmarking ML inference on H100" (any recent NVIDIA or Meta blog post)
- Read: MLPerf Inference v4 methodology paper (2024)
- Identify your 3 key measurements: (1) resume latency under load, (2) SLA compliance, (3) GPU utilization
- Sketch a test plan: baseline (vLLM only), vLLM+LRU, vLLM+Nemorix, measurement points, error bars

**Week 3: Related work + competitive analysis**
- Skim vLLM KVCache source code to understand their paging (1 hour)
- Skim NVIDIA Dynamo paper abstract + Figure 2 (routing logic) (30 min)
- Skim LMCache paper: their caching strategy (30 min)
- Skim FlexGen paper: their CPU/GPU/SSD tiering (30 min)
- Write a table: "What each system solves vs. what Nemorix adds" (2 hours)

**Weeks 4–12: Write sections + run hardware experiments**
- Write section 3 (Nemorix design): 3–4 pages, pseudocode, diagrams
- Write section 4 (implementation): 2–3 pages on C++ integration or Cython
- Do preliminary hardware runs in parallel, collect results
- Write section 5 (evaluation): 4–5 pages of benchmarks, graphs, comparison tables
- Write section 6 (related work): 2 pages
- Polish and iterate (weeks 10–12)

### Resources you need

| Resource | Cost | Why |
|---|---|---|
| Overleaf Pro (optional) | $10/month or free | Free works; pro gives more compile time |
| ICML/MLSys LaTeX template | Free | Download from conference website |
| Access to compile TeX | Free | Overleaf handles it |
| MLPerf Inference paper | Free | arXiv |
| NVIDIA Dynamo paper | Free | arXiv |
| LMCache paper | Free | arXiv (or MLSys 2025 proceedings) |
| Writing feedback from peers | Free | Ask 2–3 colleagues at imec to review sections |

### What you should NOT do

❌ Try to write the entire paper in a week. It will show.
❌ Oversell your results. Reviewers will find the gap between claims and evidence.
❌ Ignore the related work section. If you claim novelty in CXL+scheduling, cite every relevant OS paper (Belady, LRU theory, etc.).
❌ Submit without running at least one hardware experiment. Simulation-only papers are harder to get accepted.

---

## Path 2: Hardware POC at imec (6 months)

### Knowledge required

| Knowledge area | Why needed | Current level | Effort to learn |
|---|---|---|---|
| **Linux kernel / memory management** | CXL DIMMs appear as `dax` (DAX) memory; you need to mmap them or use DAX-aware APIs | Beginner (know what fork, mmap are) | 2 weeks |
| **C++ or Cython** | The simulator is Python; real deployments must be C++ or optimized Python. Profiling will require C-level understanding | Beginner (know syntax, not performance) | 3 weeks |
| **GPU driver APIs (CUDA/HIP)** | To integrate with vLLM, you need to understand CUDA memory management and device-to-host transfers | Beginner (know "GPU memory" exists) | 2 weeks |
| **vLLM architecture** | You need to hook into their KVCacheManager, understand the paging layer, and modify block allocation | Beginner (read the README) | 2 weeks |
| **NVIDIA Nsight or AMD uProf** | Hardware profiling tools to measure latency, cache hits, memory bandwidth in real time | Never used | 1 week |
| **Linux perf / flamegraph** | CPU profiling to see where time is spent (GPU wait? Memory copy? Computation?) | Never used | 1 week |
| **Docker / containerization** | To deploy the modified vLLM reproducibly on different machines | Beginner (know `docker run`) | 1 week |

### Learning path for 6 months

**Month 1: Linux + CXL fundamentals**

*Week 1–2: Linux memory management*
- Read: Linux Kernel Docs on `mm/` (memory management) — just the overview
- Read: "Understanding Linux Memory Management" (Red Hat training, free online)
- Hands-on: Write a small C program that uses `mmap` to access a large file, measure latency
- Hands-on: Clone the Linux kernel source, grep for "dax", understand how DAX works
- Expected output: You can explain what happens when you `mmap` 512 GB of CXL memory

*Week 3–4: CXL hardware basics*
- Download Samsung CMM-D or SK Hynix AiMX datasheet (ask imec for access)
- Read, especially: electrical specs, bandwidth specs, latency specs, connection topology
- Read: "CXL 2.0 Specification" (executive summary, Ch. 1–3)
- Expected output: You know the difference between CXL Type 1 (cache coherent) and Type 3 (pooled memory)

**Month 2–3: vLLM + CUDA integration**

*Week 5–6: vLLM architecture*
- Clone vLLM from GitHub: `git clone https://github.com/lm-vision/vLLM.git`
- Read: `vllm/worker/gpu_worker.py` — how does it manage GPU memory?
- Read: `vllm/kv_cache/` directory — where are the block allocation decisions?
- Hands-on: Run vLLM locally with a small model (Llama 2 7B), trace the code path
- Expected output: You can point to the exact lines where a KV block is allocated to GPU VRAM

*Week 7–8: GPU memory and CUDA basics*
- Complete NVIDIA's free CUDA course (3 hours): "Introduction to CUDA C++"
- Read: "CUDA Best Practices Guide" — focus on memory transfer and pinned memory
- Hands-on: Write a CUDA kernel that copies data from GPU VRAM to pinned host memory, measure latency
- Expected output: You understand H100 bandwidth limits and host-device transfer bottlenecks

**Month 4: Profiling and measurement**

*Week 9–10: NVIDIA Nsight*
- Install NVIDIA Nsight Systems (free, part of CUDA toolkit)
- Follow the tutorial: "Profile a simple CUDA application"
- Hands-on: Profile vLLM's inference loop with Nsight, export the timeline
- Expected output: You can identify where the 20ms+ is spent (GPU compute? Memory wait? PCIe transfer?)

*Week 11–12: Linux perf and flamegraph*
- Read: Brendan Gregg's "Linux Perf Examples" tutorial
- Install `flamegraph` tools: `git clone https://github.com/brendangregg/FlameGraph`
- Hands-on: Run vLLM, collect perf traces, generate a flamegraph
- Expected output: You can spot CPU bottlenecks like "mmap" or memory allocation overheads

**Month 5: Nemorix integration**

*Week 13–14: Design your C++ hook*
- Pseudocode: "How will Nemorix intercept a block allocation in vLLM?"
- Options: (1) Modify vLLM's `AllocateBlock()` method, (2) Create a wrapper layer, (3) Use LD_PRELOAD to intercept malloc
- Prototype in Python first: create a mock vLLM that calls your tier_manager
- Expected output: A clear architecture diagram and a small C++ proof-of-concept

*Week 15–16: Implement integration*
- Start with Cython: Convert critical path of `tier_manager.py` to Cython
- Benchmark: Cython vs. pure Python — should be 5–10x faster
- Integrate with vLLM: add your tier_manager as a shim before vLLM calls CUDA
- Expected output: A modified vLLM that uses Nemorix's eviction policy; runs without crashing

**Month 6: Experiments + validation**

*Week 17–20: Run benchmarks*
- Baseline: Vanilla vLLM, measure latency for single agent
- Baseline: Vanilla vLLM, measure latency for 10 concurrent agents (will fail or be slow)
- With Nemorix: Repeat both measurements
- Expected output: Table showing latency reduction and max agents supported

*Week 21–22: Debug and optimize*
- If Nemorix is slower than expected, profile and optimize
- If latency is higher than simulation, investigate (CXL contention? NUMA effects?)
- Document deviations from simulation

*Week 23–24: Results + video*
- Compile all results into a report with figures
- Record a 2-minute demo video showing 50 agents resuming smoothly
- Write up methodology and findings

### Resources needed

| Item | Cost | Estimated | Notes |
|---|---|---|---|
| CXL DIMM (Samsung CMM-D 32GB or SK Hynix) | €5,000–8,000 | 1× | One DIMM sufficient for POC |
| H100 GPU (via imec cluster or AWS) | Included in imec budget | 6 months access | ~€500/month if on AWS p4de instance |
| Linux server (CPU + motherboard with CXL slot) | Included in imec budget | Already exists at imec | Supermicro, Dell, or HPE with CXL slot |
| Development machine (your laptop) | Included | Already have | |
| NVIDIA CUDA toolkit | Free | Download | |
| NVIDIA Nsight Tools | Free | Download |  |
| vLLM source (GitHub) | Free | Clone | |
| Cython compiler | Free | `pip install cython` | |

### What you should learn in parallel while coding

While implementing, study:
- OS cache replacement theory (Belady, LRU, LFU) — understand optimal eviction
- GPU memory hierarchy (L1, L2, HBM3, PCIe bandwidth) — why transfers matter
- Benchmarking methodology — how to measure fairly without biasing toward your solution

---

## Path 3: Open-Source + Startup

### Knowledge required

| Knowledge | Why | Current | Effort |
|---|---|---|---|
| **Git workflow (branches, PRs, merging)** | Managing code with collaborators, not just yourself | Beginner (know `git add`) | 1 week |
| **GitHub community standards** (issues, discussions, CONTRIBUTING.md) | How to attract contributors without drowning in noise | Beginner | 1 week |
| **Licensing strategy** (MIT vs. Apache 2.0 vs. GPL) | Which license attracts users and doesn't scare VCs | Beginner | 2 days |
| **Technical writing (blog posts, tutorials)** | Explain Nemorix to engineers who won't read the paper | Intermediate (did docs) | 2 weeks |
| **Pitch deck for startups** (Sequoia format, problem-solution-market) | How to pitch to VCs in a way they understand | Beginner | 3 weeks (+ heavy iteration) |
| **Fundraising narrative** | Why investors should believe in you + your team + your market | Beginner | 2 weeks reflection |
| **Basic finance** (cap table, dilution, runway, burn rate) | Understanding VC term sheets and what they mean for you | Beginner | 2 weeks |
| **Founding team dynamics** | If you bring co-founders, how to structure equity and roles | Beginner (you're solo now) | 1 week per co-founder discussion |
| **Customer discovery (cold email, user interviews)** | How to find design partners and validate the market | Beginner | Ongoing (hours per customer, not days) |
| **Business development / sales** (if you go SaaS) | Closing the first paid customer is different from getting LOIs | Beginner | 3–6 months to competence |

### Learning path

**Month 0–1: GitHub + Open-source best practices**

- Read: "How to run a successful open-source project" (Producton Ready Microservices, Ch. 9)
- Activity: Examine 3 successful imec spin-out projects on GitHub (look at their issue templates, CI/CD, README)
- Activity: Set up your repo with:
  - CONTRIBUTING.md (how to submit PRs)
  - CODE_OF_CONDUCT.md
  - Issue templates for bug reports and feature requests
  - GitHub Actions CI/CD (auto-run tests on every PR)
  - CHANGELOG.md with release notes

**Month 1–2: Pitch deck + fundraising basics**

- Read: Sequoia's pitch deck template (search "Sequoia deck template" — it's freely distributed)
- Read: YC's "How to talk to investors" essays
- Watch: 3 successful deep-tech founder pitches (search YouTube: "pitch deck LLM inference startup")
- Activity: Write your pitch deck (10–12 slides):
  1. Problem (idle agent memory waste = $60M+ globally)
  2. Solution (Nemorix: 4-tier hierarchy + semantic eviction)
  3. Market (inference startups, cloud providers, enterprises)
  4. Unfair advantage (CXL partnerships, paper + IP)
  5. Competition (comparison table vs. vLLM/LMCache/Dynamo)
  6. Traction (GitHub stars, design partnerships, paper)
  7. Team (you, co-founders, advisors)
  8. Ask (€200K pre-seed)

**Month 2–3: Customer discovery**

- Identify 20 potential design partners:
  - 5 inference startups (Together, Fireworks, Modal, etc.)
  - 5 cloud providers (AWS AI, Azure ML, GCP Vertex)
  - 5 enterprise AI teams (from Fortune 500 list)
  - 5 academic labs running multi-agent systems
- Write a 1-minute cold email intro (not a hard sell, just "I think we solve your problem")
- Expected outcome: 3–5 meetings, 1–2 LOIs

**Month 3: Business fundamentals**

- Read: "The Lean Startup" by Eric Ries (2 weeks, foundational thinking)
- Read: YC's guide to cap tables and equity splits
- Activity: Build a simple financial model:
  - Assume 3 scenarios: conservative (10 customers/yr), base case (30), upside (100)
  - For each: calculate ARR at $50K/customer/year, runway, burn rate
  - Estimate how much money you need and for how long

**Month 4–6: Narrative building + branding**

- Write 3 blog posts:
  - "How we designed a memory manager for 1,000+ concurrent AI agents"
  - "Why CXL memory changes everything for AI inference"
  - "Open-sourcing Nemorix: benchmarks from our simulator"
- Create a 60-second demo video (no narration, just screen + music)
- Design a simple landing page: `nemorix.ai` or `nemorix-ai.github.io`

### Resources

| Item | Cost | Notes |
|---|---|---|
| GitHub Pro (for private repos during development) | $4/month | Optional; free tier mostly sufficient |
| Domain name (`nemorix.io` or similar) | ~$10/year | Hosting on GitHub Pages is free |
| Figma for graphics/logos | Free tier | Design the blog header images |
| Dev.to or Medium for blog | Free | Where ML engineers read |
| Loom for video recording | Free tier | Record your demo or tutorial |
| Pitch deck tools (Google Slides or Figma) | Free | Don't buy PowerPoint |

---

## Summary: Skills by Timeline

### **Month 1 (Now)**
Must have or learn immediately:
- ✅ Python (test-writing, profiling)
- ✅ Physics reasoning (latency calculations already in your simulator)
- 🟡 **LaTeX basics** (1 week to baseline)
- 🟡 **GitHub best practices** (1 week)

### **Months 2–3**
For all three paths:
- 🟡 **Academic writing** (reading 5 papers takes 1 week)
- 🟡 **Pitch deck** (2 weeks)
- 🟡 **Customer discovery cold email** (1 day to draft, then ongoing)

### **Months 3–6**
If pursuing hardware POC:
- 🟠 **Linux kernel DAX / CXL** (2 weeks)
- 🟠 **Cython / C++** (3 weeks)
- 🟠 **CUDA fundamentals** (2 weeks)
- 🟠 **Profiling tools (Nsight, perf, flamegraph)** (3 weeks total)

### **Months 6–12**
If pursuing startup:
- 🟠 **Fundraising narrative** (3 weeks)
- 🟠 **Sales / customer success** (ongoing learning)
- 🟠 **Finance basics** (cap tables, term sheets) (1–2 weeks)

### **Ongoing throughout**
- ✅ **Testing and validation** (you already have this mindset)
- ✅ **Technical documentation** (you already write good docs)
- 🟡 **Blog writing / communication** (3 weeks to find your voice)

---

## Reality Check: What NOT to Learn

❌ **Don't learn web development** if you're not building a web product. Nemorix is a system library, not a SaaS web app.

❌ **Don't learn Kubernetes / DevOps** unless you're running a managed service. For open-source + license deals, this is overhead.

❌ **Don't learn HTML/CSS** unless you're building a marketing website. Even then, use a template.

❌ **Don't learn advanced project management tools** (Jira, Monday.com). GitHub Issues is enough for a team of 3–5.

❌ **Don't try to learn everything at once.** You have 12 months. Batch the learning by quarter.

---

## Recommend: Learning order for maximum impact

### If you have 3 months of full-time (you do this full-time):

1. **Week 1–2**: Paper writing skills + LaTeX (enables publication path)
2. **Week 3–4**: GitHub + community standards (enables open-source path)
3. **Week 5–6**: Pitch deck + customer discovery (enables startup path)
4. **Week 7–12**: Parallel paths based on what resonates

### If you have 12 months part-time + imec resources:

1. **Month 1**: Paper (weeks 1–2) + GitHub (weeks 3–4)
2. **Month 2–3**: Hardware POC setup (Linux + CXL basics)
3. **Month 4–5**: Hardware POC implementation (C++ / Cython + profiling)
4. **Month 6**: Pitch deck + customer outreach
5. **Month 7–12**: Paper revision + fundraising conversations

---

## Shortcuts: What you can outsource

| Task | Outsource to | Cost | Why |
|---|---|---|---|
| LaTeX typesetting | Overleaf + template | Free | Templates exist; you just fill in content |
| GitHub repo setup | Copy imec's spin-out template | Free | They have a standard CONTRIBUTING.md + CI/CD |
| Logo / branding design | Fiverr or Canva | $50–200 | Not worth your time; you're not a designer |
| Patent filing | imec's TTO or a patent lawyer | €2,000–5,000 | Necessary; imec subsidizes 50% |
| Pitch deck design | Use Sequoia's template; don't customize heavily | Free | Good templates > custom ugly design |
| Blog platform setup | GitHub Pages + Jekyll | Free | Forget Medium, own your content |
| Video editing | Raw Loom recordings + music | Free | Don't make it Hollywood-polished; authenticity wins |

---

## Final advice

**You don't need to learn everything now.** You have:
- ✅ Strong fundamentals in the hardest part (algorithms, testing, physics)
- ✅ A proven ability to write clear documentation
- ✅ The discipline to validate claims with tests

What you need to learn next depends on which path you choose **first**:

- **Choose paper + hardware first** if you want credibility and to work with imec's resources
- **Choose open-source + design partners first** if you want to validate the market quickly
- **Choose pitch deck first** if you're confident enough to start fundraising

All three paths are open. Pick the one that excites you, spend 2–3 weeks learning the baseline skills, and then build. You'll learn the rest by doing.
