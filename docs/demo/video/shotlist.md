# demo-1min-v2.mp4 shot list

1920x1080, 30 fps, H.264, 60.0 s, no audio. Captions are burned in; the same text is in `captions-v2.srt`.
Every shot was re-recorded at 4:44 PM on 2026-10-17 from the live dashboard (127.0.0.1:5173) and the DoneTwin replays (127.0.0.1:8000) using Playwright. No live agent runs were started. The Stage "Four layers" slide is left out until its inference figure is fixed.

| Time | On screen | Caption |
|---|---|---|
| 0:00 to 0:08 | Overview hero with the four live layer tiles | Thesis; one harness, four layers, Nemotron on Crusoe |
| 0:08 to 0:20 | Overview, scrolls to the "The loop" diagram | Solve (Shared Brain), Review (You.com, numeric claims), Gate (exact verifier) |
| 0:20 to 0:32 | DoneTwin replay `162447-9b6f`, small-10000 hall: re-planning, then "Collision. Column C1." in red | Every Nemotron layout failed the collision check and was rejected. Hall: no gain yet |
| 0:32 to 0:46 | Stage tab, rack placement replay `qaplib-dc-research-2`: KEPT and REJECTED cards with claimed vs measured gain and the You.com sources | 4 of 6 kept; 0.597 v0 cold to 0.845 final, 6 unseen instances |
| 0:46 to 0:56 | Overview layer tiles | Inference 31% less per correct answer; QAPLIB 3.6% from best-known; GPU kernels on a Mac GPU, correct 20% to 60%, none faster than PyTorch |
| 0:56 to 1:00 | End card | Bring a verifier. Get a team that improves. |
