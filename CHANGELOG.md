# Changelog

All notable changes to Super Processor are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [4.5.2] - 2026-11-26

### Changed

- A scene with DPIR keeps the temporal half of its look's `hqdn3d`. DPIR
  cleans each frame on its own and replaces only the spatial half, so asking
  for more denoise no longer makes the scene flicker a little more between
  frames.

## [4.5.1] - 2026-11-26

### Fixed

- A note that asks for more denoise in two ways, such as "still noisy, more
  denoise", now raises the scene's DPIR level by one step, not two. One note
  moves the level one step at most; its last request decides the direction.

## [4.5.0] - 2026-11-26

### Added

- DPIR, a stronger GPU denoiser (DRUNet), for the scenes you ask it for. A
  note on a scene's preview such as "more denoise", "still noisy", or "too
  grainy" turns it on for that scene, one level of 3 per note; "less
  denoise" lowers it and "no DPIR" turns it off. A result note does the
  same for the scenes in its time range. Gemini cannot turn it on or off.
  It runs before SPAN, in place of the look's FFmpeg denoise, and the
  preview's "What this scene gets" says when a scene has it.
- `doctor` says where the DPIR weights are, or that they are fetched on
  first use.

### Security

- The DPIR weights (`drunet_color.pth`, about 130 MB, MIT) are fetched only
  from one pinned KAIR release URL, over HTTPS, from GitHub's hosts (each
  redirect is checked too). The download goes to a private temp file and
  is kept only if its size and sha256 match; it stops at the expected size.
  Loading checks the hash again on the bytes it loads, with
  `weights_only=True`. `SUPER_PROCESSOR_DPIR_WEIGHTS` uses your own copy and
  is never downloaded over; `SUPER_PROCESSOR_DPIR_DOWNLOAD=0` turns the
  download off.

## [4.4.0] - 2026-11-25

### Changed

- `super-processor review` now serves the web app itself. A normal
  `pip install ancient-processor` includes the built app, so the printed
  address opens the full UI with no Node or Vite needed. The Vite dev
  server and the GitHub Pages shell still work for development.
- `super-processor review JOB_ID` opens that job in the web app. A job made
  with the command line (no wizard state) is pointed to
  `segment --accept/--note` instead.
- The release workflow can be started by hand as a dry run: it builds and
  checks the web app, the sdist and every wheel, and publishes nothing.

### Removed

- The built-in server-rendered wizard pages and the older per-job
  split-review page. The JSON API and every wizard step are unchanged.

### Security

- The served app has a strict Content-Security-Policy (its own scripts,
  styles, fonts and connections only, no framing). Actions take the token
  only in the X-Wizard-Token header, never from a form field or the
  address. Media from the served app no longer carries the token in its
  address, the Host header is parsed strictly (one header, matching port),
  and a decoy cookie set by another local port can no longer lock the app
  out. Static files are served only from the built app, by name, with
  fixed content types.

## [4.3.3] - 2026-11-24

### Changed

- Previews no longer draw boxes over faces, hands and text, and the list of
  what a scene gets no longer says what is handled gently. Those areas are
  still found and still restored more gently, so renders are unchanged.
  This applies to the web app and the built-in page.

## [4.3.2] - 2026-11-23

### Security

- Replies from Gemini, and the Gemini API's own answers, are treated as
  untrusted throughout. Deeply nested, oversized, non-finite or
  wrongly typed replies now end in a clean error or bounded values
  instead of crashing the worker, looping without end (a model list that
  never stops paging) or spending a minute of CPU on 50,000 segments.
  Text from Gemini is kept to one capped line, keys outside the reply
  schema are no longer applied or replayed, odd model names and file
  addresses are refused, and damaged job files give a clear error. A new
  test feeds hostile replies into every Gemini call.

## [4.3.1] - 2026-11-22

### Security

- The wizard now needs its token on every page, API call and video or still,
  not just on changes. `super-processor review` prints an
  `open in your browser: http://127.0.0.1:PORT/?token=…` address; opening it
  logs the browser in with a cookie and removes the token from the address
  bar. Anything else gets 401, so other programs or users on the same
  computer can no longer read the token, list your jobs, download your
  videos or replace your Gemini key.
- A page on the Vite dev or preview port (5173, 4173), such as another
  project's dev server, no longer gets the wizard's token or any state
  without it.
- Another website can no longer show the wizard's videos or stills inside
  its own page: a cross-site load needs the token.
- The split review page of `review JOB_ID` has a token too and can no longer
  be shown inside another site's frame.
- A connection that stays silent is closed after 60 seconds, and at most 64
  are served at once, so idle connections can no longer tie up the wizard.
- Every wizard response tells the browser not to frame it, not to guess file
  types, and not to cache it.
- The web app only connects to a wizard on this computer (`127.0.0.1`,
  `localhost` or `[::1]`). A link with `?api=` pointing anywhere else is
  ignored with a message, so it cannot collect your Gemini key or video.
- The web app takes the token from the address the CLI prints
  (`#token=…`), keeps it for the tab and sends it with each request.
- The web app refuses to run inside another site's frame, so its buttons
  cannot be clicked through a hidden frame.
- Your jobs are private to you on Linux and macOS. `~/.super-processor`,
  the jobs folder, each job folder and the uploads folder are now readable
  by you only (0700), and the wizard's state, the Gemini chat log and
  uploaded videos are 0600, so other accounts on a shared computer can no
  longer list or copy your videos, outputs or Gemini notes. Folders an older
  version left open are tightened the next time you start. A folder you
  choose with `--jobs-dir` keeps its own mode, but the jobs inside it are
  private, and a folder that every user may change is refused.
- Only real video files can be picked. A playlist (`.m3u8`) or an FFmpeg
  concat script disguised as an `.mp4` is refused with a clear message in
  the wizard, `job probe` and `diagnose`, so a crafted download can no
  longer make the app read, upload to Gemini or restore another video from
  your disk. FFmpeg may also no longer open anything but local files for
  any input.
- On Windows, `ffmpeg`, `ffprobe`, `nvidia-smi` and the file dialog's tools
  are taken from PATH only, and PowerShell and `taskkill` from `System32`,
  so an `ffmpeg.exe` sitting in the folder you start the program from is
  never run in place of the real one.
- The small copy of your video and the stills sent to Gemini no longer carry
  the original file's metadata or chapters: a phone video's GPS location,
  title, author, comments and recording time stay on your computer.
- `super-processor doctor` no longer prints API keys. It shows only that a
  key is set and its last 4 characters (in the text report and in
  `--json`), shows `SUPER_PROCESSOR_LLM_BASE_URL` without any user name,
  password or query, and now also says where the Gemini key comes from.
- The Gemini key is sent in the `x-goog-api-key` header instead of the
  address of each request, and a key with a space or line break in it is
  refused on the setup screen. No error message, progress line, saved
  `wizard_state.json`, API answer or page can show the key any more.
- "Replace key" on the setup screen takes effect at once; before, the wizard
  kept using the old key until it was restarted.
- The copy uploaded to Gemini is deleted from Google's File API as soon as
  nothing needs it: when the render starts, when you start a new video, and
  when you replace the key. Before, it stayed there for about 48 hours.
  Going back to the scenes afterwards uploads it again.
- Gemini answers holding `NaN` or `Infinity` are refused, so a bad answer can
  no longer turn part of the render black or break the result screen.
- The SPAN weights are now loaded from the exact bytes whose checksum was
  checked. Before, the file was checked and then opened again, so a file
  swapped in between (for example while a render waited for the GPU) could
  be loaded.
- The gpu extra now needs PyTorch 2.6 or newer. Older versions let a crafted
  weights file run code even in the safe loading mode (CVE-2025-32434).
  The README now creates the models folder private to you.
- The "install the gpu extra" hint now names the real package,
  `pip install "ancient-processor[gpu]"`. Before, it said
  `super-processor[gpu]`, a different PyPI name that could install someone
  else's code.
- Every GitHub Action in CI, Pages and the release is now pinned to a fixed
  commit instead of a tag or branch the action's owner could move. The
  GitHub Release is created with GitHub's own `gh` tool instead of a
  third-party action, so no outside code holds the token that can write
  releases.
- The PyPI upload token and the release-writing token are now given only to
  the one job that needs each, and the PyPI job runs only for `v*` tags in
  the `pypi` environment. CONTRIBUTING.md lists the GitHub settings to turn
  on by hand (protected `main`, a reviewer on `pypi`, a `v*` tag rule).
- The Pages build, including every pull-request run, now has a read-only
  token. Only the deploy step, on pushes to `main`, can publish the site.
- Release builds no longer fetch the newest setuptools, Cython,
  setuptools-scm, build, cibuildwheel, twine or pytest at tag time. They are
  exact versions installed from files with checksums, and the build uses
  only those, so a bad new release of one of them cannot end up in the
  wheels published to PyPI. The `wheel` package is no longer needed.
- Dependabot now also watches the web app's npm packages and the pinned
  release tools, and keeps the pinned actions up to date.

## [4.3.0] - 2026-11-22

### Changed

- The web app has the redesign's look: a top bar with the job's title, a
  connection pill and All jobs; a sidebar with the video's still, its facts
  and the five steps; and self-hosted Source Sans 3, Source Serif 4 and
  JetBrains Mono fonts. On each step change the heading takes focus and the
  tab title names the step, and run progress is read out at milestones.
  Below about 900 px wide the sidebar sits above the step.
- The web app's first screen is one Welcome & setup page in place of the
  introduction, the LLM choice and the Gemini key steps: "Choose a video",
  what you need (with whether a Gemini key is found, and Replace key) and
  what leaves your computer. The page shown with no wizard connected has
  the same look, with the connect form beside it.
- When the web app can't reach the wizard, the step stays on screen dimmed
  with its buttons turned off, under a "Can't reach the wizard" banner with
  Try again and Change address; anything typed is kept, and the page looks
  again every few seconds until the wizard answers.
- Choose a video has the redesign's drop area ("Browse this computer…",
  "Type a path instead", or a file chosen in the browser), and a file
  dropped or chosen in the browser gets a file card with its frame, size,
  picture size and length checked before "Find the scenes". A file the
  wizard refuses shows which check failed and why, with "Choose another
  video".
- Finding the scenes shows the time left (or spent), one bar, and its
  stages (stills, the silent copy, the upload, Gemini) with their times,
  saying so when Gemini works from the stills instead; "You can close this
  tab", the notify switch and Cancel sit under it. When Gemini's quota is
  used up, its split failed or it returned no scenes, the Scenes step says
  so with "Try again" (after the wait Gemini gave) or "Ask Gemini again",
  and "Use the whole video as one scene".
- The Scenes step has the redesign's layout: "Gemini found N scenes" with
  whether Gemini watched the video or worked from stills (and why), the
  timeline with a handle on each cut ("Drag to move this cut") and a time
  ruler, and a row per scene with its still, times, length and what is
  handled gently (faces, hands, text, fine pattern). Merge, split and move
  cut are under each scene's Edit. On the right, "Suggest different cuts"
  chips (merge, split, fewer or more scenes) and your own words go to
  Gemini with "Ask Gemini to re-split", above "Looks right — preview
  scenes".
- A scene's preview is one large comparison of the real before and after
  clips, kept in step: Wipe (drag the handle or use the arrow keys), Side by
  side or A/B, one play button and scrubber, frame steps and a 100% view,
  with the protected boxes labelled. Keys: Space plays, B flips, ←/→ step a
  frame, Z zooms. Beside it: Gemini's check, what the scene gets in plain
  words, AI detail Off or on, "Something off?" chips that add to your note
  for "Redo this preview", and Advanced settings with the raw numbers. The
  button names the next scene ("Approve scene 3 — next: Night street"); a
  preview Gemini couldn't check is marked "Check this", with Try Gemini
  again and Approve anyway.
- The web app's Render step has the redesign's look. Ready to render
  offers the size as two cards, "Sharper and bigger" and "Same size,
  cleaned up", with the real output size and a still from your previews,
  beside your scenes with their AI detail and Start render. While it
  renders it shows the time left, one bar, the render's stages, "Safe to
  close this tab" and Cancel render, and the tab title shows the percentage.
  A failed render marks Render in red, says why, and offers Try the render
  again and Render without AI detail.
- The Result step is "Watch the restored film": the restored video in the
  viewer with the scene seams and any possible seam drawn under it, and
  "Keep this video". "Fix part of it" marks a part from the player (Mark
  start / Mark end, or I and O while it plays), adds fixes as chips (less AI
  detail, smoother, sharper, fix a seam) and your words, shows the note
  Gemini reads, and re-renders that part. "Changes so far" lists the notes
  behind each version. "Start over from the scenes…" moves into a More menu
  with "Restore another video", behind a dialog that says what is deleted
  and what is kept.
- After "Keep this video" the page says "Your restored video is ready",
  with the file's name, picture size, length and size, Copy link, Save a
  copy, a still of one scene before and after, and the versions. All jobs
  shows each job's status beside its button, and Refresh and Disconnect
  move into its More menu.
- The screens shown while a scene is looked at, and when a run was
  interrupted, have the same look as the other steps. A progress screen
  shows and reads out one number, the bar's; the line under the bar names
  the stage without its own count. Finding the scenes shows the time spent
  in large type and the stage's time left beside it. The Scenes step keeps
  "Looks right — preview scenes" on screen while its list scrolls, and
  says "Your video has N scenes" once the list has changed. All jobs says
  Render for a job whose scenes are all approved. The Welcome page no
  longer offers a local model, which can't be used yet.

## [4.2.0] - 2026-11-21

### Added

- The wizard's `/api/state` reports the last scan, preview or render as
  `run` (its kind, job, start and end times, and whether it finished,
  failed or was stopped), so a page can tell a crash from a success even
  after the step has moved on.
- An approved scene can be opened again from a scene preview or from
  "Ready to render" (`/api/scene` with its `index`), up to the first scene
  not approved yet. Approving it goes back to the furthest scene reached,
  so earlier approvals are not lost, and the overall progress bar does not
  go back while it is looked at again. `/api/state` now has a `job`
  summary for a stepper and scene strip: the job's title, each stage and
  whether it is done, and each scene with its still, whether it is
  approved, previewed, and can be reopened.
- Finding the segments and the render can be stopped: a Stop button on
  their pages, or `/api/cancel`. The run stops within moments (the GPU
  can take up to a minute) without waiting for Gemini to answer, and goes
  back to where it started: the segments from before (or the pick screen),
  "Ready to render" with every approved scene kept, or the previous result
  with its settings. A stopped job is never marked failed. A render that
  has started replacing its output can no longer be stopped. A stop asked
  for just before the wizard was closed still applies when it restarts,
  and a job left mid-step can be stopped without starting its work again.
  `/api/state` has `cancellable`, `cancel_requested` and `stopped`.
- The scenes can be changed by hand on the Segments step: move the cut
  after a scene, join a scene with the next one, or split one
  (`/api/segments` with `action` `move`, `merge` or `split`, the scene's
  `index`, and `at` in seconds). The limits are the same as for Gemini's
  proposals: at least 3 seconds a scene and at most 10 scenes. Each change
  sends the `revision` the page shows, so a second window showing older
  scenes is refused instead of overwriting them. A change that leaves the
  scenes as they were keeps their approval. `/api/state` shows the
  segments' `revision`, `approved`, `duration_s`, `min_segment_s`,
  `max_segments` and each scene's `index`.
- A list of every wizard job, newest first (`/api/jobs`): its title (the
  name of the file as it was picked), a still from the video, whether it
  has a result, was left mid-step, or is busy in another wizard window on
  the same computer. `/api/jobs/open` with `job` opens one where it was
  left. A job's result and stills can be fetched by its id
  (`/jobs/<id>/output.mp4`, `/jobs/<id>/thumb.jpg`), and `output_url`
  now names the job, so a page left open never plays another job's video.
  A job is only shown as busy in another window while the process that
  started its work still runs: a wizard that crashed, or a restart, never
  leaves the job locked, even when its process id is later reused.
- Every result is kept. Each finished render, and the result a revise or
  a "start over" replaces, is saved in the job's `outputs/` folder as
  `v001.mp4`, `v002.mp4`, … with the note that asked for it, and listed
  in `/api/state` as `versions` on the Result and Done steps (playable at
  `/jobs/<id>/outputs/v001.mp4`). Nothing is deleted automatically. On
  Linux and macOS a version takes no extra disk space; on Windows it is a
  full copy, shown as "Keeping a copy of this version". A version that
  cannot be kept (a full disk) is a warning on the result, never a failed
  render; a "start over" that cannot keep the result is refused instead, so
  the result is never deleted unkept. Each version lists the scale and
  strength it was rendered with, and a render resumed after reopening the
  job is kept with the note that asked for it.
- `super-processor review --prefetch` previews the next scene in the
  background while you look at the current one, so approving a scene
  usually shows the next preview at once. It is off by default, since it
  spends GPU time and Gemini quota on scenes you may redo or never reach.
  `/api/state` shows it as `segment_preview.next`, and a redo, leaving the
  scenes, or a render drops it along with its files, without waiting for
  a Gemini answer it was still expecting. It never uploads the video to
  Gemini; it uses an upload only while one is still live.
- The web app has a Stop button while it finds the segments and while it
  renders. It turns into a disabled "Stopping…" until the stop lands, and
  the step it goes back to shows a plain note saying what was kept. A job
  left interrupted asks Resume or Stop and says what each one does, and a
  render that failed shows its error with a "Try the render again" button.
- The web app shows the job's title and a five-step stepper (Video,
  Scenes, Previews, Render, Result) with a short line under each step,
  such as "2 of 5 approved". From the previews on, a scene strip lists
  every scene with its still, time range and state (Approved, Reviewing,
  Waiting) below the step, scrolling sideways on a phone, and an approved
  scene can be opened again with "Look again", which moves focus to the
  scene's page. The job's percentage never goes back while a step runs,
  nor when a scene is looked at again.
- The Scenes page has a timeline for changing the cuts yourself: drag a
  cut, focus it and move it with the arrow keys (Enter saves it), or type
  its time with "Move cut…"; merge a scene with the next one, or split one
  at a time you type. A click or tap on a cut only selects it. It says at
  once when a change would leave any scene under 3 seconds or make more
  than 10 scenes. If another window changed the scenes first, the page
  shows them as they are now and says why, even when that takes it to
  another step.
- The web app has an "All jobs" page, "Your videos", reachable from the
  top bar on every step and shown first when the wizard would open on the
  file pick and jobs exist. Each job shows its still, title, file name,
  when it last changed and where it is ("Result ready", "Scenes · waiting
  for you", "Render · stopped partway", "Busy in another window"), with a
  link to watch its result. Opening one switches the wizard to it where
  it was left; a job stopped partway opens on the Resume or Stop choice
  and never starts work by itself. A job another window is working on,
  or any other job while a run works here, cannot be opened and says why.
  "Restore a new video" keeps the open job where it was.
- The Result and Done pages list every kept version, newest first, with
  when it was kept, its scale and size, the note that asked for it, and
  Play and Download buttons; the one the result is now is marked
  "Current". While the scenes are found or the render runs, "Notify me
  when it's done" sends a desktop notification if the tab is in the
  background when it finishes or fails (the browser asks permission only
  when you tick it, and the choice is remembered). With `--prefetch`, a
  scene preview says when the next scene is being prepared in the
  background.

### Changed

- A job left mid-step (the wizard was closed while it found the segments,
  looked at a scene or rendered, or the job was reopened from the list) no
  longer starts that GPU or Gemini work again by itself. The page asks
  first: Resume carries on, Stop goes back to the step before it.
  `/api/state` says `interrupted`, `?run=1` waits until a POST to
  `/api/resume`, and `/api/cancel` stops it instead.

### Fixed

- After new scenes are approved (a note to Gemini, a manual edit, or
  starting over from the result), the previews no longer show the old
  scenes' cached video: preview URLs now carry the plan's generation.
- Stopping the wizard server in the middle of a render or a scene scan now
  ends its FFmpeg and SPAN work too, instead of leaving it writing to the
  job's work files. The job keeps its step, and the next start asks
  whether to resume it.
- "Ready to render" refuses to start while a scene has changes that have
  not been previewed yet, and says which scene.
- A half-written or damaged `shot_plan.json` no longer breaks `/api/state`
  on a scene preview or "Ready to render"; the error is shown instead.
- A render now replaces `output.mp4` once, with a finished file that
  already has its audio. A render that failed while copying the audio (or
  a whole-video revise that failed partway) used to leave a silent new
  picture, or an old picture no longer marked for fast revises; the
  previous output is now kept exactly as it was.
- The wizard's scene list, shot list, plan and Gemini chat are now saved
  the same safe way as its state: through a temp file of their own that
  replaces the old file in one step, retried while Windows briefly holds
  the file open. A page refresh racing a save no longer fails the request
  or reads a half-written file, and no `.tmp` file is left behind.
- Two Gemini calls on one job finishing at the same moment could drop
  each other's turns from the saved chat, or leave a question without its
  answer. Each question and answer are now saved together under one lock,
  so the chat Gemini is shown on the next call is always complete.
- A wizard button pressed just as a scan, preview or render started (or
  just as it finished) could be applied in the middle of that work and
  be overwritten by it. Actions and background work now take turns, and
  the wizard counts as busy until the work has fully wrapped up.
- A failed first render no longer throws the job away. The wizard goes
  back to "Ready to render" with the error and every approved scene kept,
  so pressing Render again retries it; the job is no longer marked failed.
  A render started from the result screen's segments that fails now puts
  back the settings the existing result was made with, and a failure after
  the new video is already in place (such as the seam check) shows that
  new video rather than the old settings.
- A page left open on an old job could change the job another window had
  since opened. Such a request is now refused with "this job is no longer
  open in this window".
- A stopped render or Gemini copy ends its FFmpeg (on Windows, the whole
  process tree) within moments instead of leaving it encoding, removes its
  half-written `.encoding.mp4` or part file, and releases the GPU, so the
  next preview or render can start at once.
- The web app no longer waits forever on "Finding the segments", a scene
  preview or the render when the step was saved but its work never began:
  after two checks that show nothing working on it, it starts the work
  again. A job left interrupted is never started this way; it still waits
  for Resume or Stop. Every action from the web app now names the job the
  page shows, so a tab left open on an old job cannot change another one.

## [4.1.1] - 2026-11-17

### Fixed

- `diagnose` suggested brightening and adding contrast to nearly every dark
  video, turning black backgrounds milky grey and burning out whites. It
  pushed each picture's average toward mid grey and counted every black
  pixel as clipped: a 1990s documentary's interview lit against black got
  contrast 2.0 and +0.44 brightness. It now judges 9 frames spread over
  the whole video and corrects only a faded picture: one with no true
  black in any frame but a title card or two (black at level 20 or above),
  measured inside any letterbox or pillarbox bars. Its black goes back to
  video black, and its whites are stretched as far as a fade of that lift
  explains (at most 2 times), never dimmer than they were and with no
  sampled frame pushed into clipping. A picture that reaches true black is
  left alone however dark or dim it is, since a dim transfer cannot be told
  from a film shot dark on purpose. Full-range video (yuvj, MJPEG) gets its
  own brightness, levels are corrected before white balance, and a video
  of unknown length is not judged from its first second. Four archive
  films are now gently corrected, every modern clip and the documentary
  are left alone, and faded copies come back close to their originals.
  Still missed: a faded film whose samples land mostly on black title
  cards. Diagnosing takes about 0.1 s longer at SD and 0.6 s at 1080p.

## [4.1.0] - 2026-11-16

### Fixed

- `diagnose` suggested stabilising, at its strongest setting and a 30%
  zoom, for nearly every video. It measured how much the picture changed
  between two frames, so any movement, fade or cut counted as shake: 11 of
  12 test clips came out at the maximum. It now reads 8 frames in a row at
  each sample point (every second frame at 50-60 fps), finds the
  whole-frame shift between neighbours up to 5% of the width, skips cuts
  and repeated frames (24p film carried at 30p), and measures how much the
  shift changes from frame to frame, scaled to 30 fps, so a steady pan
  scores nothing and the same shake reads the same at 24, 30 or 60 fps.
  The zoom follows how far the picture strays (1-15%), and vid.stab's
  shakiness is rounded instead of cut down. Every real test clip now stays
  off; simulated handheld shake on a 1947 film, on 360p and on 25-60 fps
  video is caught. Slow sway under about 1 Hz and a subject filling half
  the frame are not told apart from a still camera. Diagnosing takes about
  0.3 s longer.

## [4.0.1] - 2026-11-16

### Changed

- Releases are published to PyPI as `ancient-processor`
  (`pip install ancient-processor`). The installed command is still
  `super-processor` and the Python package is still `super_processor`.
  4.0.0 was only released on GitHub.

## [4.0.0] - 2026-11-15

### Added

- `SUPER_PROCESSOR_SPAN_WEIGHTS` points at the SPAN weights file when it is not
  at `<models>/SPAN/2xLiveActionV1_SPAN_490000.pth`. The file is checked by
  size and sha256 before it loads, and a wrong or missing one names the
  pinned download.
- The `gpu` extra installs spandrel, which loads the SPAN network.
- `THIRD_PARTY_NOTICES.md` credits the SPAN weights and architecture.
- While a render runs, the progress screen suggests undervolting the GPU once
  if the render has slowed by at least 20% from its first minute while the
  GPU is at 80 C or more or the driver is slowing it for heat. A SPAN render
  keeps an RTX 5090 near 500 W, which heats the whole case: a 20-minute 1080p
  film took the CPU to 92-95 C while the blend and encode still ran on it.
  The app never slows itself down for heat. GPU readings come from
  `nvidia-smi`.

### Changed

- The last step of a render copies AAC and MP3 source audio onto the
  upscaled picture as it is, instead of encoding it to AAC again, and keeps
  the copy only if its audio decodes cleanly (a recording whose audio
  changes format partway is still encoded). Other audio is encoded to AAC
  as before. On a 20-minute film the re-encode was 15 of the step's 18
  CPU-seconds, a single-core burst that took a heat-soaked CPU from 86 C to
  94.6 C after the GPU had finished; the step now takes 4.3 CPU-seconds.
- SPAN (2xLiveActionV1_SPAN, Apache-2.0) replaces FlashVSR as the only
  upscaler. In a bake-off on an RTX 5090, FlashVSR v1.1 blended at strength
  0.5, as the wizard uses it, was almost the same as plain SPAN 2x on SD
  (consistency 41.9 against 41.5 dB, detail 32.1 against 32.6, added shimmer
  +0.04 against +0.07), at 97 against 3 minutes per 20-minute film and 26
  against 2 GB of GPU memory. On 1440x1080 FlashVSR did not fit 32 GB at all;
  SPAN does 2880x2160 at about 30-40 fps in about 4-9 GB. Recent jobs used
  strength 0.5 or less for 85% of segments, never above 0.7, and never 4x.
  Strength still means the same: the plain upscale's shapes plus that share
  of SPAN's fine detail.
- Every source the wizard accepts (up to 1080p) is offered 1x and 2x, and
  segment previews render at the plan's scale.
- Each clip streams through SPAN in one pass, with the per-frame work on the
  GPU. FFmpeg decodes with NVDEC (`-hwaccel cuda` when a cached probe shows
  it works; codecs NVDEC cannot read fall back to software decoding), applies
  the look's clean filter, and pipes `yuv420p` frames to a reader thread that
  fills blocks of about 128 MiB. On the GPU each block goes from YUV to RGB
  (BT.601), through SPAN, into the strength blend with a Lanczos-3 base and
  a Gaussian detail split (per-pixel strength from a protected-region map
  built once per clip), and back to `yuv420p`. A writer thread pipes the 2x
  frames into the delivery encoder, which adds only a look's grade and grain
  on the CPU. Nothing is written to scratch disk; an earlier chunked SPAN
  pipeline wrote about 63 GB of lossless 4K chunk files for a full-length
  1080p film and ran a 16-core CPU at its 95 C limit. On an RTX 5090, 1500
  frames of 1440x1080 rendered to 2880x2160 at 26.8 fps end to end with 0.054
  CPU-seconds per frame, and the CPU went from 70 C to 76 C over the minute.
  A stream that blended (lanczos, Gaussian blurs, mixes at 4K) and encoded
  with x264 on the CPU ran at 22.2 fps and 0.43 CPU-seconds per frame, half
  of it in x264 and a quarter in the blend; the GPU path's average Y, U and V
  are within 0.3 levels of it, at 44.96 dB PSNR. Segment previews use the same path under
  one model load, without the 8 s limit on a batch.
- Pass-through spans (no SPAN, or scale 1) are one encode from the source:
  NVDEC decode, the look's clean step, a lanczos resize, the finish step,
  and the delivery encoder. They went through a 2x `libx264` veryfast work
  file and then a 4K FFmpeg blend whose blurs and lookup tables a strength of
  0 threw away. On 200 frames of 1440x1080 with NVENC: 2.6 s and 0.027
  CPU-seconds a frame, against 5.0 s and 0.26, for identical frames.
- Every output is H.264. The legacy CLI export (`plans --choose`, recipe
  runs) uses `libx264` `medium` CRF 23 by default (40.63 dB in 4.0 MB on 120
  frames of 1440x1080, where the `libx265` CRF 24 it replaces made 40.60 dB
  in 3.4 MB), or `h264_nvenc`, `h264_qsv`, `h264_amf`, or
  `h264_videotoolbox` with `--encoder`. The `encode_hevc_size_cap` op is now
  `encode_size_cap` and the export codec is `h264`; recipes saved with the
  old names, or with HEVC encoders, load as the H.264 equivalents.
- Doctor no longer requires `libx265`, lists the H.264 hardware encoders,
  and reports whether `h264_nvenc` and CUDA decoding actually run here.
- A failed NVENC or CUDA-decode probe is asked again after 60 s (a working
  one is kept), and each render decides its encoder once, so one output
  never mixes encoders. Probes no longer pipe their output, which on Windows
  could hang a timed-out probe behind Chocolatey's ffmpeg shim.
- The output of an upscale holds exactly the source's frame count: a decoder
  up to 8 frames (or 2%) short repeats the last frame, extra frames are
  dropped, and a larger shortfall stops the render rather than splice a
  misaligned picture.
- Restoration delivery stays H.264 only and is encoded with `h264_nvenc`
  (preset `p5`, tune `hq`, VBR CQ 17, High profile) when a one-frame test
  encode, cached per FFmpeg binary, works on this machine, and with `libx264`
  preset `faster` (instead of `medium`), CRF 16, otherwise (CI, Macs, AMD).
  Both keep `yuv420p`, an IDR every second, and no B-frames, and every part
  of a job uses the same encoder so parts still join by stream copy. Against
  a lossless 2880x2160 reference (200 frames), NVENC CQ 17 gave 81.8 MB,
  44.29 dB PSNR and SSIM 0.98588 with almost no CPU; x264 `faster` CRF 16
  gave 88.7 MB, 44.44 dB and 0.98655 for about 48 CPU-seconds. The delivery
  marker records the encoder used (`codec`), its quality setting, and
  version 3. Before a revise splices into an output from an earlier version
  (3.7.0 or earlier) or one made with the other encoder, that output is
  re-encoded once into the current settings, so one file never mixes
  encoders. NVENC is given H.264 level 4.1 or the level the size and rate
  need: left to pick the lowest, it held a 320x180 clip at level 1.1 to
  192 kbit/s (26.9 dB on moving grain, against 41.8 for x264). Outputs
  outside 145x49 to 4096x4096, which NVENC's H.264 refuses, use `libx264`.
- When the GPU runs out of memory, the upscale retries with smaller batches,
  down to one frame, and then stops with advice to close other GPU programs.
  Retries add no joins.
- A plan saved at 4x loads and renders at 2x. A result note on a job whose
  output is still 4x renders the whole file again at 2x, and the Result line
  reports the size on disk.
- The Ready screen offers "Upscale 2× · SPAN on the GPU", and the per-segment
  toggle reads "Use upscaling" / "Skip upscaling".
- Render progress counts the frames an upscaled part has handed to the
  delivery encoder, since the upscale and the encode now run at once.

- The timing check that decodes every frame of a source before it is
  restored now uses 8 decoder threads instead of 1: 42 s instead of 203 s for
  a 20-minute 1440x1080 film, with byte-identical timestamps. All 32 threads
  took 23 s but ran the CPU to 94 C.

### Removed

- FlashVSR v1.1 and FlashVSR+, with `SUPER_PROCESSOR_FLASHVSR`,
  `FLASHVSR_HOME`, and `FLASHVSR_PLUS_HOME`.
- 4x output, and with it `SUPER_PROCESSOR_GPU_VRAM_GB` and the GPU-memory
  check that gated it.
- The system-RAM fallback and its CUDA managed-memory allocator
  (`managed_alloc.cu`, built with `nvcc`), and the chunk-halving retries.
- The chunked upscale: 8 s chunks with a 0.5 s overlap and their lossless
  (`-qp 0`) work files in `flash_chunks/`.
- The 1280 px limit on upscaled sources and the Ready screen's notes about
  source size and GPU memory.

### Fixed

- Look estimation measured noise as the whole picture's brightness variance,
  which is mostly contrast, so it asked for full-strength denoise on nearly
  every clip, clean ones included. It now measures grain as the median
  difference between neighbouring pixels, across and down, on a square of
  up to 256 px cut from the middle of the picture at the source's own pixels
  (scaling to 160x90 first averaged the grain away). On three SD archive
  films and four 480p and smaller modern clips the old measure asked for
  strength 0.22-1.0 every time; the new one leaves them all alone (grain
  sigma 0-1.9), and asks for denoise only on clips with heavy grain or phone
  sensor noise (sigma 5-5.5, strength about 0.45).
- A damaged file now gives a clear error instead of a traceback: invalid,
  non-UTF-8 or too deeply nested JSON, or an unknown step, in a wizard state
  file; a truncated `samples.json` or segments file; a segment missing
  `index`, `start_s`, `end_s`, `keyframe_s` or `look_group`, or holding
  `NaN`, `Infinity` or an out-of-range number; and a Gemini chat file whose
  root is not an object or whose turn has no role.
- Gemini's file expiry (nanoseconds with a `Z` zone) was read as local time;
  it is now read as UTC, and an offset such as `+05:30` is kept.
- A shot check with index `true` was applied to shot 2; it is now refused.
- The pick page said videos are upscaled up to 1280 px on the long edge; it
  now states the wizard's real limit, up to 1080p and 30 minutes.

## [3.7.0] - 2026-11-12

### Added

- `SUPER_PROCESSOR_FLASHVSR=plus` restores with FlashVSR+ turbo (colour
  correction on) instead of FlashVSR v1.1, which stays the default. It needs
  about 60% less GPU memory, so 4x of 854x480 fits in 32 GB without the
  system-RAM fallback, but it is slower at 2x. `FLASHVSR_PLUS_HOME` points at
  the checkout; missing code or weights are named before anything runs.
- `SUPER_PROCESSOR_FLASHVSR=auto` uses FlashVSR+ for 4x clips and v1.1 for
  2x, or the other when only its checkout folder is present. Replaying a trailer plan at the
  wizard's 2x strengths gave the same picture with v1.1 in 64% of the time;
  at 4x, v1.1 needs the slower system-RAM fallback.
- Gemini watches the video. The wizard uploads an audio-free copy
  (`gemini_upload.mp4`: long edge at most 960 px, `libx264` CRF 28, source
  frame rate) to the Gemini File API, which samples it at 1 fps. Segments
  are proposed from the whole video and each segment's settings from its
  stretch of it. Google deletes the upload after about 48 hours;
  `gemini_video.json` records it so later requests reuse it while it lives.
  If the copy or upload fails, Gemini works from the still grid as before and
  the Segments screen says why. Video requests wait up to 300 s, skip a busy
  model without retrying, and stop the model failover on a timeout.
- Gemini chooses the output size (1x, 2x, or 4x) while it proposes segments,
  and the Segments and Ready screens show its reason. 2x stays the default. 4x is offered
  only for sources up to 960 px on a GPU with at least 24 GB, read from
  `SUPER_PROCESSOR_GPU_VRAM_GB`, `nvidia-smi`, or a loaded PyTorch; on a small
  source the Ready screen says when the GPU is too small for 4x.
- Both shells show the whole job's percentage ("N% of the job") and the
  running step's time left ("about 4 min 05 s left"), or the time spent while
  Gemini is thinking. `/api/state` progress adds `overall`, `eta_s`,
  `elapsed_s`, and `updated_at`. The estimate ignores model loading and
  blends the step's average rate with its recent one; the React shell counts
  it down between polls.

### Changed

- Render progress is weighted by work: a frame FlashVSR restores counts 20
  times a frame that is only resized, so a pass-through span no longer makes
  the percentage and time left jump. The message now reads "N% of the work".
- Segment previews still render at no more than 2x now that a plan can start
  at 4x, so they stay out of the system-RAM fallback, and a 1x plan runs no
  FlashVSR for its previews.
- FlashVSR v1.1 chunks no longer pass through a lossy CRF 20 file: the
  restored frames go straight to the lossless chunk, as FlashVSR+ already
  did. While the GPU restores a chunk, a reader thread cuts and loads the
  next one and a writer thread encodes the last. On a 60 s 640x480 clip
  at 2x this took 214 s instead of 250 s, and the chunks gained 43-45 dB
  of PSNR over the old CRF 20 copies.

### Fixed

- A source whose long edge is over 1280 px is no longer offered FlashVSR.
  Gemini chose 2x for a 1440x1080 film; its 2880x2160 output did not fit a
  32 GB GPU, and the system-RAM fallback put the 20-minute render at about
  16 hours. Such a source now keeps its size (clean-up and finishing only),
  Gemini is not asked for a scale, and the Ready screen says why. A plan
  without Gemini's choice starts at 1x there instead of 2x.
- A result note changes the output size only by naming a multiple, such as
  "make it 4x", and only one the Ready screen offers for the source. A note on a
  "Don't upscale" render used to re-render the whole clip at 2x with FlashVSR,
  and a 1080p source could reach 4x. When Gemini answers a different scale, the
  Result screen says so and how to ask.
- A note on a "Don't upscale" render that names no size is refused with a
  message, since strength cannot change the picture without FlashVSR, and the
  Result line says "FlashVSR off" for that render.
- After a scale change fails to render, the plan goes back to what `output.mp4`
  holds, so later notes no longer fail on a size mismatch.
- The Result player shows the revised video after a revise. The URL stayed
  `/output.mp4`, so a browser could keep playing what it had buffered from the
  previous render, even though the file on disk was correct.
- `npm run preview` serves the production build at `/super-processor/` again;
  the page was blank because the preview server used base `/`.
- A refused host-memory allocation (torch's "can't allocate memory", or a
  `MemoryError` subclass) now counts as out of memory, so the clip is retried
  in shorter pieces instead of failing outright.

- The Result line describes the whole render instead of its last span. A
  render that ended on credits passed through without FlashVSR read
  "strength 0.00" although every restored segment ran at 0.30–0.45; it now
  reads "strength 0.30–0.45 · FlashVSR off for 9.9 s". The JSON API adds this
  as `strength_summary`, which the React shell shows; `strength` is unchanged.
- CI runs the offline frontend regressions and production build before packaging.
  The manifest regression skips in wheel-only environments without development
  build tools instead of adding setuptools as a runtime dependency.
- Preserve the latest requested revision range and knobs separately from split
  render spans, including after rendering and reloading a job, so follow-up
  result notes receive the complete previous range. Older plan JSON still loads.
- Batch timing preflight honors an engine's explicit FFmpeg binary, with an
  optional caller override, instead of requiring a separate ffprobe on PATH.
- Normalize packaging-test fixture paths to native separators so manifest
  exclusions are checked correctly on Windows as well as POSIX systems.
- FFmpeg cancellation escalates after a one-second grace period when a process
  ignores SIGTERM; the runner owns signaling and reaping to avoid delayed PID
  races. Progress reads FFmpeg's `out_time_ms` as microseconds.
- Doctor requires `libx264` for wizard previews/delivery as well as `libx265`
  for legacy HEVC export. Requirements now distinguish current wizard filters
  from legacy scene detection, and routine test instructions exclude live Gemini
  calls with separate opt-in quota/privacy guidance.
- Source distributions retain the React HTML entry point while excluding Cython
  annotation pages. Source/wheel metadata and a React build from an unpacked
  source distribution are checked during development.
- Frontend integration: Vite dev and preview proxy segment/shot stills and
  preview videos alongside API/uploads and output. Direct upload CORS preflight
  permits `X-File-Name` while preserving the local-origin allowlist.
- Long-pass requests and progress polls time out after 10 seconds and retry
  transient failures with bounded backoff. Exhausted connections show an explicit
  retry action that refreshes server state/token without cancelling server work.
  Shared runners retain all active progress subscribers across effect cleanup
  and StrictMode re-subscription, keyed by origin and route; tokens are scoped
  by origin rather than shared across backend connections.
- Document the printed ephemeral backend URL, `VITE_PROXY_TARGET`, and direct
  connection setup. Add offline Node frontend regressions for dev/preview media
  forwarding, subscriptions, token isolation, retries and timeouts, plus backend
  upload preflight regressions. Browser E2E remains a separate acceptance check.
- Seam checks share the renderer's coalesced, frame-snapped parts and nominal
  FlashVSR chunk ranges. Check retained joins (first at 8 s, not the 7.5 s
  overlap start), including end-anchored chunks; respect `use_upscale` and
  scale-1/pass-through parts. Range revisions check actual keyframe-expanded
  splice edges and replacement joins before pending state is cleared. Synthetic
  regressions cover visible jumps, equal spans, toggles, fractional frame rates,
  and revision edges. Thresholds are unchanged; runtime OOM/custom/fallback
  chunk joins are not recorded and remain a documented coverage limitation.
- Result-note revisions preserve each shot's FlashVSR/pass-through choice,
  look, and protected regions, including notes spanning multiple shots. A
  cross-shot revision re-renders the timeline rather than splicing one shot's
  settings across its neighbours.
- Keep the whole-shot restoration strength cap when any flagged face, hand,
  or text kind lacks a valid box, even if another kind was boxed successfully.
  Shots with valid boxes for every flagged protected kind keep their requested
  strength outside the boxes.
- Refuse VFR/unsupported restoration timing before wizard job creation, GPU
  processing, or replacing an output. Admission now checks the complete decoded
  presentation timeline and final frame duration, rather than trusting the
  `is_vfr` rate-metadata heuristic. Shared render, revise, preview, and engine
  entry points protect resumed jobs and direct callers too; successful checks
  are cached by source identity and modification metadata. External CFR
  conversion is an explicit workaround that may change frames/timing, not an
  automatic normalization or full VFR support.
- `review JOB_ID` resumes the requested wizard job instead of the last active
  job. Reopened review windows keep their own session, so another window cannot
  redirect their media, approvals, or revisions to a different job.

## [3.6.3] - 2026-10-26

### Fixed

- Segment previews run at 2× again, on the GPU. The preview plan started at
  the largest scale the source allows, and Gemini, offered 2× or 4×, picked
  4×. At 854×480 that is 3456×1920, which FlashVSR cannot fit in 32 GB of
  VRAM even 21 frames at a time, so every preview went to the system-RAM
  fallback: in the trailer run each one took about 33 GiB of RAM for about
  four minutes, and with a game holding VRAM the first one, before the
  fallback had limits, crashed the computer. 4× is still offered on the
  Ready screen, which now starts at 2×.

## [3.6.2] - 2026-10-26

### Fixed

- The wizard no longer keeps about 10 GiB of RAM between previews. Loading
  FlashVSR's weights and decoding the frames go through several GiB of host
  memory; once freed it stayed in glibc's heap, split over one arena per
  worker thread. After every FlashVSR call the heap is now trimmed back to
  the system: across five trailer previews the wizard sits at 2.5 GiB
  between calls, against 5.3 GiB before (10.4 GiB in the live wizard). The
  peak during a call, about 10 GiB, is the upstream loader reading the
  weights and is unchanged.

## [3.6.1] - 2026-10-26

### Fixed

- The system-RAM fallback can no longer run the computer out of memory. A
  preview next to a game paged about 80 GiB into RAM and the kernel began
  killing programs. Three limits now hold it:
  - The child works in the shortest pieces (17 frames) instead of starting
    again at 8 s, since FlashVSR's memory grows with the frames in a call.
  - Its allocator stops at a byte limit: the free VRAM plus what RAM can be
    spared, which is what is available past a reserve of a quarter of the
    RAM (at least 8 GiB), and never more than half the machine. Past the
    limit it returns its cache to the driver and then refuses, which ends
    in the usual out-of-memory message. It throws to refuse: PyTorch does
    not check a null block, which came back as a tensor at address 0 and
    broke the CUDA context.
  - The wizard stops the child if available RAM falls below half that
    reserve, whatever the cause.
  With too little RAM to spare, no child is started. The allocator's cache
  also reuses any block up to twice the size asked for and returns itself
  to the driver once more than 2 GiB sits idle: reusing only exact sizes
  had left 13.6 of 27.6 GiB idle for one 72-frame preview. That preview,
  with 8 GiB of VRAM free, now peaks at 22.3 GiB held and leaves 53.6 of
  78.9 GiB of RAM available, in 45 s.

## [3.6.0] - 2026-10-25

### Added

- FlashVSR uses system RAM as well when the GPU is too full. If even the
  shortest piece (17 frames) runs out of GPU memory, the clip is restored
  once more in a child process that uses CUDA unified memory: what fits
  stays on the GPU and the driver pages the rest to system RAM. It is slower,
  not different: at 2x of 854x480 with 16, 12, or 8 GiB free, every clip
  finished at 0.43-0.70 s a frame (0.27-0.30 s with the GPU to itself) and
  matched the normal output's SSIM. Its allocator keeps freed blocks for
  reuse like PyTorch's own; asking the driver for each block made it three
  to four times slower. It is built with `nvcc` on first use; without the
  CUDA toolkit, or off Linux (Windows drivers do not let managed memory
  grow past free VRAM), the run fails as before, with the same advice.

### Changed

- At most 10 segments, each at least 3 s, and fewer is better. Gemini is
  asked for that, and the repair holds it whatever the reply: while there are
  too many segments or one is too short, the shortest is merged into its
  shorter neighbour, keeping the longer one's label and what both contain.
  Gemini's 22 segments for the Sintel trailer (several only 1 s long) become
  10 of 3.9-6.9 s with every face, hand, and text flag kept. The minimum was
  0.5 s, and there was no maximum.

- FlashVSR fits next to other programs on the GPU. Its working memory grows
  with the frames in one call (at 2x of 854x480: 15.5 GiB for 16 frames,
  16.9 GiB for 24, against 3.4 GiB of weights), so a clip that runs out of
  memory is retried in pieces of half the length, down to 17 frames. Each
  finished piece goes to disk before the next starts, so memory follows the
  piece, not the clip. With a game holding 9 GiB, a 130-frame clip now
  finishes in 10 pieces instead of failing. Pieces are planned in whole
  frames, and the last one is full length so no remainder falls below what
  FlashVSR accepts. Keeping the weights in RAM was measured too: it saves
  only about 0.5 GiB, so it is not used.

### Fixed

- A failed FlashVSR call no longer keeps GPU memory. Its traceback held the
  frames' tensors and the pipeline, and the error raised from it kept that
  alive, so a preview that ran out of memory left 13 GiB taken until the
  server restarted, and the retry failed too. The traceback is now cleared
  before the memory is released, and the pipeline's own caches are cleared
  after every clip.

## [3.5.2] - 2026-10-25

### Fixed

- Every segment lines up with the source frame for frame. Parts were cut
  half a frame before their first frame; with the time written to six
  decimals that sits on FFmpeg's rounding edge, and in a file that keeps one
  tick per frame (the Sintel trailer) 141.5/24 rounded down, so 4 of its 14
  segments started a frame early and the seam check flagged the fade at
  0:09.8. Parts are now cut a quarter frame early. FlashVSR also returns up
  to three frames short of a clip, and the last frame used to be repeated in
  their place; clips now get that many copies of the last frame before
  FlashVSR, which are trimmed off after, so the tail is real frames.
- A look no longer pushes black below video black or white above video
  white. FFmpeg's `eq` scales around the middle of 0-255, so the trailer's
  credits look (contrast 1.10, brightness +0.02) took black from 16 to 7.5,
  and black visibly changed at the cut into it. The finish step now ends by
  holding luma to 16-235 and chroma to 16-240, after the grain; black now
  stays at the source's 16.5 across that cut.
- The final audio mux no longer drops frames. It used `-shortest`, so a
  source whose audio ends before its picture (the Sintel trailer: 51.95 s of
  audio, 52.21 s of picture) lost its last frames after the frame-count check
  had passed. The mux now runs exactly as long as the picture, cutting longer
  audio there, and keeps the picture without audio if the frame count would
  still change.
- FlashVSR no longer crops the picture. Its loader centre-crops the upscaled
  frame to whole 128-pixel blocks, so an 854x480 clip at 2x came back as the
  middle 1664x896 of 1708x960, and the delivery blend laid that over the full
  frame squeezed to the same size, out of line toward the edges. Each clip is
  now padded with its own edge pixels to whole blocks before FlashVSR and
  cropped back after, so the output is exactly the source times the scale and
  lines up with it (SSIM against the source 0.97, was 0.87, on a trailer
  clip).

## [3.5.1] - 2026-10-24

### Changed

- The Gemini key pasted on the setup screen is saved to `credentials.json` in
  the user's config folder (`~/.config/super-processor` on Linux,
  `~/Library/Application Support/super-processor` on macOS,
  `%APPDATA%\super-processor` on Windows; `SUPER_PROCESSOR_CONFIG_DIR`
  overrides it), readable only by the user, and used on every later run. It
  used to live in the server process and, best-effort, the OS keyring, so a
  restart could lose it. `GEMINI_API_KEY` still wins; an old keyring entry is
  read last and never written. The setup screen says where the key in use
  comes from, and a key that cannot be saved is an error instead of silently
  lasting only until the next restart.

### Fixed

- Tests no longer touch the real keyring or config folder. A test that saved
  a key used to overwrite the user's keyring entry with `abc123`.

## [3.5.0] - 2026-10-23

### Added

- Go back and forth between Setup, File, and Segments. Those steps in the rail
  are links when they can be reached, and each screen has a back link
  (`POST /goto`). Going back keeps the job: approving the same segments again
  carries on at the segment you were on.
- The file step opens the desktop's own file dialog ("Choose a file…",
  through the XDG portal on Linux) and uses the real path, or takes a video
  dropped on the page, which is streamed into `<jobs>/uploads/` and removed
  again if it cannot be used. Typing a path moved under "Type a path
  instead".
- Upscaling is a choice on the ready screen: 2×, 4× when the source is small
  enough, or "Don't upscale" (scale 1: same size, clean-up and finishing only,
  no FlashVSR). Each segment preview can also turn FlashVSR on or off for
  that segment; later Gemini proposals keep the editor's choice.

### Changed

- The step rail drops Upscale: Setup · File · Segments · Previews · Result.
- A file that cannot be read says so in one line, naming the file, instead of
  showing FFprobe's output.

## [3.4.0] - 2026-10-23

### Changed

- Gemini errors read as one line. The API's error body (several kilobytes of
  JSON with help links and quota metrics) becomes what went wrong and when to
  try again: "daily quota used up; try again in about 16 h", "the API key was
  rejected; replace it on the setup screen", "this model is no longer
  available". When several models fail the same way, the failover names them
  together once.
- The Segments screen shows a timeline bar with one block per segment, as
  wide as its share of the video, and each segment with the sampled still
  nearest its middle, its time range and length, and what it contains as
  tags. The approve button says how many segments it approves.
- When Gemini could not split the video, the Segments screen says so instead
  of "Gemini split the video", and the reason sits in a notice box; the
  segment preview shows its warnings the same way.
- The pick screen describes the segment loop instead of a whole-file pass.
- The React shell matches all of the above.

### Added

- `GET /segment_stills/t_<ms>.jpg` serves the still grid to the Segments
  screen; other names and symbolic links are refused.

## [3.3.2] - 2026-10-23

### Changed

- The render's progress follows the work instead of jumping between fixed
  steps. Each part counts for its share of the source frames, FlashVSR
  reports after every chunk of about 8 s, and the bar reads "Upscaling on
  the GPU: N% of the frames". `render_plan`, `render_shots`, and
  `apply_range_revise` take an `on_progress` callback, and `UpscaleRequest`
  carries one to the engine.

## [3.3.1] - 2026-10-23

### Changed

- `docs/system-design.md` describes the segment loop: local still sampling,
  repaired Gemini segments, one preview per segment with a seeded window,
  optional pass-through spans, the full restart that keeps the chat, the
  shared `segments.json` name, and the job files and steps that go with
  them.

## [3.3.0] - 2026-10-23

### Added

- Full restart from the result screen. A note sent with "Start over from the
  segments" (`restart=1` on `POST /api/result`) clears the job's segments,
  plans, previews, output, and work folders, keeps `gemini_chat.json` with
  the note added, and goes back to sampling. Gemini then proposes segments
  and settings again with the whole earlier exchange in view. A note without
  `restart=1` still retunes one time range.

## [3.2.0] - 2026-10-23

### Changed

- The wizard now runs the segment loop: pick → segments → one preview per
  segment → ready → upscale → result. Gemini proposes segments from a local
  still grid; the editor approves them or sends a note. Each segment is then
  planned, previewed, and checked on its own, and approved or redone with a
  note before the next. A ready screen lists every segment's settings before
  the full render. `SUPER_PROCESSOR_SEGMENT_LOOP` is gone.
- Both the HTML pages and the React shell show the new steps, with a progress
  bar on the busy ones. The step rail reads Setup · File · Segments ·
  Previews · Upscale · Result.
- Gemini's segments say what each contains (faces, hands, text,
  fine_pattern), so those segments are boxed and held down as shots were.

### Removed

- The shot flow's wizard steps (`finding_shots`, `shots`, `planning`,
  `looks`), their routes (`/api/scan`, `/api/shots`, `/api/plan`,
  `/api/looks`), and shot merge and split in the wizard. A job saved on one of
  those steps goes back to the pick step with a message.

## [3.1.0] - 2026-10-23

### Added

- Spans and previews marked `use_upscale: false` skip FlashVSR. The clip is
  resized with lanczos and goes through the same delivery encode at strength
  0, so it keeps its look, matches its neighbours' size, and the joined
  output still holds every source frame once. Full renders, range revises,
  and previews all follow it, and `upscale_plan.json` spans carry the flag.

## [3.0.0] - 2026-10-23

### Added

- The segment loop runs in the wizard when `SUPER_PROCESSOR_SEGMENT_LOOP=1`
  is set. After the pick, one worker samples the stills and asks Gemini for
  segments (`GET /api/segment?run=1`). `POST /api/segments` approves them or
  sends them back with a `note`. Each segment is then planned, previewed,
  and checked on its own (`GET /api/analyze?run=1`); `POST /api/preview`
  approves it and moves to the next, or redoes it with a `note`
  (`resample=1` picks a new preview window). When every segment is
  approved, `POST /api/ready` with `action=render` starts the full render.
  `/api/state` returns `segments`, `segment_preview`, and `segment_index`.
  The page shell does not show these steps yet.
- Each segment's preview is a random window of up to 3s inside it, seeded
  by job, segment, and preview count, and kept across redos.

## [2.9.0] - 2026-10-23

### Added

- Wizard steps for the segment loop (`sampling`, `proposing`, `segments`,
  `analyzing`, `preview_segment`, `ready_to_render`) and a `segment_index`
  in the wizard state. Nothing leads to them yet; the page shell shows a
  placeholder on them.
- Worker progress. The scan, planning, and render workers record
  `{phase, pct, message}` in the wizard state as they go, and `/api/state`
  returns it as `progress` while a worker runs.

### Changed

- The wizard state file is schema 3. Schema 2 files load unchanged; a file
  older than schema 3 on the removed flow's `analyzing` step still goes back
  to the pick step.

## [2.8.0] - 2026-10-23

### Added

- Gemini proposes the segment loop's segments from the still grid and its
  times (`GeminiClient.propose_segments`). A re-proposal carries the
  editor's note, and the saved chat is replayed so Gemini sees what it
  proposed before. `repair_coverage` turns any reply into segments that
  cover the clip once: gaps are split down the middle, an overlap is cut
  where the later segment starts, and slivers under 0.5s are folded in.
- Gemini can set `use_upscale: false` on a shot's look so it skips FlashVSR.
  The setting is kept in `shot_plan.json`; rendering follows it from a
  later release.

## [2.7.1] - 2026-10-23

### Fixed

- The FlashVSR runner test skips when `ffmpeg` is missing from `PATH`, so
  the release wheel tests pass. The 2.7.0 release built no wheels and was
  not published; 2.7.1 ships its changes.

## [2.7.0] - 2026-10-23

### Security

- The wizard makes a random token each run and prints it after the URL.
  Every POST and `?run=1` request must carry it (`X-Wizard-Token` header or
  `token` field); the wizard's own pages and `/api/state` hand it out, and
  pages refuse to be framed. Another site can no longer drive the wizard in a
  browser that omits `Origin`. This replaces the `replace=1` key guard, which
  any page able to post could also send.
- A junk or negative `Content-Length` now gets 400 instead of dropping the
  connection.

### Changed

- Picking a video through a symbolic link works again; the link is followed.
- Docs: the GitHub Pages shell needs
  `SUPER_PROCESSOR_ALLOWED_ORIGINS=https://lkpjohn2026.github.io` to drive a
  local wizard.

### Added

- Segment loop groundwork. `segment_loop.py` samples a still grid from the
  source (about one a second, at most 120, always the first and last frame)
  into `segment_stills/t_{ms}.jpg` inside the job folder, and keeps the
  wizard's segment list in `segments.json` (schema 2, `kind: "wizard"`). The
  older segment review refuses to read or overwrite that file, and the
  wizard refuses the older review's. The wizard does not use the loop yet.
- Shimmer check. Each preview's fine-detail layer is measured for how much
  it changes frame to frame relative to how much there is, against the
  source. A preview whose texture is over 1.6x less stable than the source's
  fails as `flicker` even when Gemini passed it, loses 0.15 strength, and is
  redone; Gemini also sees the number. The Looks screen shows it.
- Seam check. After every render, each place FlashVSR restarted (chunk
  boundaries, boundaries between unlike shots, revise edges) is compared
  with the source; a frame jump 2.5x beyond the source's is reported on the
  Result screen with a ready-made "smooth" note. Results are in
  `render_report.json`.

- Protected regions. Before planning, Gemini boxes the faces, hands, and text
  in each shot that has them, from stills at its start, middle, and end. In
  the delivery blend the FlashVSR share is held to 0.3 inside face and hand
  boxes and 0.15 inside text boxes, feathered so there is no seam. The Looks
  screen draws the boxes over the previews. A shot whose boxes could not be
  placed falls back to the whole-shot 0.6 cap.

- Per-shot looks (loop 2). After the shot list is approved, Gemini proposes a
  strength and look for each shot (and one scale for the video), each shot
  gets a short before/after preview, and Gemini checks the previews for
  identity changes, bad anatomy, garbled text, waxy skin, oversharpening,
  halos, fake texture, colour shifts, and softness. A failed shot gets
  corrected settings and a new preview. The new Looks screen shows each pair;
  the editor approves all or writes a note that redoes one shot. Strength is
  capped at 0.6 on shots with faces, hands, or text, and 4x is only offered
  for sources up to 960 pixels on the long edge. Without Gemini the recipes
  come from the measurements. New steps `planning` and `looks`, routes
  `/looks`, `/api/plan`, `/api/looks`, and `/previews/`.
- `FlashVsrEngine.upscale_many` restores several short clips with one model
  load; the previews use it.

- Shot review (loop 1). After a file is picked, FFmpeg finds the cuts
  (`scdet`), measures each shot in its own pass (blockiness, blur, noise,
  brightness, contrast), and saves a still per shot. Gemini labels each shot
  from its still and numbers, using fixed lists of problems and contents. The
  new Shots screen lets the editor merge or split shots and approve the list;
  approval writes one upscale span per shot. If Gemini cannot label the
  shots, the list keeps the measurement hints and says why. New steps
  `finding_shots` and `shots`, routes `/shots`, `/api/scan`, `/api/shots`,
  and `/shot_stills/`.

- Per-shot looks. Each span in `upscale_plan.json` now carries a `look` of
  bounded FFmpeg settings applied in a fixed order around FlashVSR: clean
  (`deblock`, `denoise`) before it, finish (`contrast`, `brightness`,
  `saturation`, `gamma`, `grain`) after it. Cleaning first gives FlashVSR
  fewer compression blocks and less noise to sharpen into made-up texture.
  The plain-upscale base of the strength blend is cleaned the same way.
  Plans without a `look` load as neutral, so existing jobs render as before.
- A plan with several shots renders each shot on its own and joins them by
  stream copy. Shot edges snap to source frames, and the join is checked to
  hold exactly the source's frame count. Neighbouring shots with the same
  settings render as one part, so FlashVSR loads once per run of alike shots.

### Changed

- Gemini stays on the legacy `generateContent` API. Each call resends the
  stored chat as `contents`, and a model switch repeats that full request
  against the next model. The Interactions API is not used.

- The strength blend now takes shapes, colour, and layout from the plain
  upscale of the source and only fine detail (below a 1.5 source-pixel
  Gaussian) from FlashVSR. FlashVSR can sharpen what is there but can no
  longer move, reshape, or recolour it, at any strength. Strength 1.0 is no
  longer "FlashVSR alone"; it is the source's structure with all of
  FlashVSR's fine detail.
- The 0.6 strength cap on shots with faces, hands, or text now applies only
  when no protection boxes were placed.
- A scale change on part of the clip keeps every shot's strength and look
  and only changes the scale, instead of collapsing the plan into one span.
  A result note keeps the look of the shot it names.

### Fixed

- The wizard no longer traps the user after Done. Done and Result have a
  "Start a new video" button (`POST /new`, `/api/new`) that keeps the old job
  on disk and returns to the pick screen, or to setup when no Gemini key is
  available. Before, `super-processor review` reopened the same Done page.
- A first upscale that fails (no CUDA, missing FlashVSR weights, a bad file)
  marks the job failed and returns to the pick screen with the error, instead
  of showing a Result page with no video. A failed range revise still keeps
  the previous `output.mp4` on the Result screen.
- Intro copy, the static shell tour, and the system design no longer claim
  "no invented frames, faces, or scenes" or that FFmpeg applies the repairs.
  They now say FlashVSR adds fine detail, strength sets how much, and every
  frame and its timing are kept.

### Removed

- The wizard's Gemini split / FFmpeg enhance flow: the analyze, overview,
  split choice, and per-segment enhance steps, their pages and API routes
  (`/analyze`, `/overview`, `/split`, `/segment`, `/previews/`, stills), the
  segment-choices render, and Gemini's split and enhance calls and schemas.
  Since the switch to FlashVSR, picking a file goes straight to the upscale,
  so none of it was reachable. A saved wizard state on one of those steps now
  loads as the pick step. The CLI `diagnose`, `plan`, `segment`, and `plans`
  commands are unchanged.

### Added

- Local FlashVSR restoration upscale on the wizard picture path. FFmpeg trims,
  splices, and copies audio. Gemini turns a result note into a time range,
  scale, and strength, then the cited range is restored from the original source.

### Fixed

- The wizard serves only localhost: a foreign `Host`, a foreign `Origin`, or
  a cross-site request that would start work is refused with 403. CORS answers
  only the wizard itself, the Vite dev/preview ports, and the published React
  shell; add origins with `SUPER_PROCESSOR_ALLOWED_ORIGINS`.
- Upscale and analyze passes run on one worker thread. `/render?run=1` and
  `/api/render?run=1` start a pass and return at once; a reload or a second
  tab no longer starts a second GPU pass on the same job. Posts are refused
  while a pass runs. FlashVSR loads are serialized process-wide.
- A result note that changes scale on part of the video re-renders the whole
  clip at that scale instead of splicing two resolutions into one stream.
  A same-scale note trims the spans it overlaps instead of dropping them.
- Range revises no longer re-encode the parts of the video the note did not
  name. The upscaled picture is encoded once with a keyframe every second and
  no B-frames; a revise snaps out to those keyframes and stream-copies the
  head and tail. Work clips handed to FlashVSR are lossless.
- A Gemini call that times out moves on to the next candidate model instead
  of failing the request. The per-model read timeout is 60 seconds (was 120).
- Upscale strength now does what it says. It is the share of the FlashVSR
  picture in the output; the rest is a lanczos upscale of the same frames,
  mixed in the delivery encode. 0.0 invents nothing, 1.0 is FlashVSR alone.
  Before, strength only switched FlashVSR's attention settings, so every value
  from 0.0 to 0.54 produced the same output. FlashVSR now always runs at its
  upstream settings.
- Every FlashVSR clip is checked against the frame count and frame rate of
  the clip it was given. Up to 8 frames (or 2%) of difference at the end, or
  a rounded fps, is conformed losslessly; more raises before anything is
  spliced. A range revise that would change the output's total frame count
  is refused and the previous output is kept.
- `output.mp4` and preview clips are streamed with HTTP Range support (206,
  416, HEAD) instead of being read whole into memory, and sent with
  `Cache-Control: no-store` so a revise is not hidden by the browser cache.
  The final mux writes the `moov` atom first (`+faststart`).
- The wizard refuses inputs above 1080p (1920x1080 in either orientation) or
  longer than 30 minutes at the pick step, before a job is created.

### Planned

- Wire a real local LLM path behind the deferred stub screen (OpenAI-compatible
  or native multimodal endpoint).
- Chunked FlashVSR execution for clips up to about 30 minutes.

## [2.6.0] - 2026-10-03

### Added

- Localhost website UI for the wizard: polished server-rendered screens with
  step rail, LLM type choice (Gemini live + deferred local stub), lettered
  split/enhance options, and the full sketch flow through result/happy.
- CI HTTP test drives the wizard website from introduction through done
  (FakeTransport Gemini + FFmpeg), asserting each screen’s copy and step rail.

### Fixed

- Wizard tests that encode tiny clips skip when `ffmpeg` is missing from
  `PATH` (cibuildwheel / headless release environments).

## [2.5.0] - 2026-10-03

### Added

- Enhance **Something else** loop: `combine_enhance_revise_message` /
  `revise_enhance` feed the user note plus prior structured options back into
  Gemini; the wizard stays on the segment until an option is previewed and
  accepted (tests cover five revise rounds).

## [2.4.0] - 2026-10-03

### Added

- Localhost Gemini wizard via `super-processor review` (no job id): setup,
  pick file, structured split choice, per-segment options with short previews,
  enhance-and-concat render, and a happy/revise result step.
- Gemini structured-output client (`gemini.py`) with chat persistence,
  dynamic-FPS sampling, allowlisted ops validation, and failover across
  multimodal models with usable RPM.
- Design docs: revised [docs/system-design.md](docs/system-design.md) and new
  [docs/ui-wizard.md](docs/ui-wizard.md).
- CI job **Gemini live API** calls the real Gemini structured-output endpoints
  using the repository secret `Gemini_API_Test` (mapped to `GEMINI_API_KEY`).

### Changed

- Wizard product path uses Gemini for split/diagnosis; classical CV estimators
  are not used by the wizard. Export keeps source aspect (no vertical social
  default).
- Matrix unit tests run with `-m "not gemini_live"` and never receive the
  Gemini secret; live split/enhance/multi-turn checks run only in the dedicated
  job.

### Fixed

- Gemini API key storage works without an OS keyring backend (headless CI).
- Live Gemini calls close HTTP error bodies, retry transient 503s, and skip
  image-generation models when discovering candidates.
- Windows CI retries Chocolatey FFmpeg installs and skips ffmpeg-dependent
  wizard tests when the binary is missing from `PATH`.

## [2.3.1] - 2026-09-26

### Fixed

- Hardware-decode sample tests pass an explicit ffmpeg path, so wheel
  builds no longer fail when the image has no FFmpeg binary.

## [2.3.0] - 2026-09-26

### Added

- `doctor` lists `hevc_cuvid`, `hevc_qsv`, `hevc_amf`, and
  `hevc_videotoolbox` next to the hardware encoders.
- `segment JOB --decode NAME` reads sample frames with that decoder. One-hertz
  rows keep the same shape. A decoder this ffmpeg does not have fails before
  sampling.
- A frame the device cannot decode is read again in software. The sample row
  and the segment label match a software read.

### Fixed

- The hardware-encoder doctor check and `plans --choose` test no longer
  require `ffmpeg` on `PATH`. Wheel builds were failing when the image had
  no FFmpeg binary.

## [2.2.0] - 2026-09-26

### Added

- `doctor` reports `hevc_nvenc`, `hevc_qsv`, `hevc_amf`, and
  `hevc_videotoolbox` as present or missing. `libx265` stays required.
- A recipe may name one of those encoders. Any other codec name fails
  validation. The default remains `libx265`.
- `plans JOB --choose N --encoder NAME` renders with that encoder's own
  rate-control flags. Trim and the filter graph stay the same. A missing
  device fails with the doctor line before any segment is encoded.

## [2.1.0] - 2026-09-26

### Added

- `review JOB` serves a localhost split page. It lists each segment's range,
  context, problem, and still. Accept and a note call the existing `segment`
  command.

## [2.0.0] - 2026-09-26

### Added

- A named fault filters one segment's treatments. A sentence maps onto that
  fault list only after a fault rejection. Neighbors keep the treatment they
  already shared.
- `plans JOB --choose N` is the approval to render. Each kept segment is
  encoded with libx265, audio is cut on the same timestamps, and the parts
  are concatenated.
- The size-cap bitrate uses the sum of the kept durations.

### Fixed

- A split of an oversized range stays at least 5 seconds from both ends, so a
  loud change at the edge cannot leave a piece shorter than 5 seconds.

## [1.3.0] - 2026-09-26

### Added

- A fixed treatment list for each segment problem. Silhouette lifts shadows
  harder than low light. Strong denoise cannot share a treatment with sharpen,
  and steps stay in allowlist order.
- Five timeline plans. The first is the highest score. Later plans change as
  many look groups as they can, and neighbors that share a look share a
  treatment. `plans JOB` writes `plans.json` and prints each assignment.
- A three-second preview window around each key frame, clamped inside that
  treatment's trim, and one cached encode of that window.

## [1.2.0] - 2026-09-26

### Added

- Split loop for a local file: one-hertz sample rows, segments of 5–120
  seconds (at most 15, files over 30 minutes refused), a context and problem
  label, one still per segment, and shared look groups.
- `segment` prints a proposed split. `--note` re-splits inside those bounds
  (too many, too few, a boundary move, or a relabel). `--accept` records the
  timeline; a second accept leaves the job unchanged.
- Sharpen and trim on the recipe allowlist. Sharpen stays inside a mild
  unsharp range. A trim must keep at least 5 seconds.

## [1.1.0] - 2026-09-20

### Added

- Classical saliency-aware 9:16 reframe (edge-energy subject tracking) with
  planned crop windows wired into FFmpeg templates.
- Structured LLM recipe patches (`{"patches":[...]}`) outside `--safe-mode`,
  still clamped to allowlisted ops/params.
- Optional OS keyring lookup for API keys via `keyring` when installed.
- Doctor checks for `libx265`, vid.stab / `deshake`, and optional `libvmaf`.
- `--acknowledge-size-risk` on diagnose / plan / apply / job reframe so a
  below-floor size cap becomes an explicit warning instead of a hard error.
- Live encode progress on stderr during preview and apply.

### Changed

- Package classifier moved from Pre-Alpha to Alpha.
- Local `.dogfood/` artifacts are gitignored.

## [1.0.0] - 2026-09-20

### Added

- End-to-end product loop: diagnose → plan → preview → apply → show.
- CV look/motion estimators, recipe validation, FFmpeg templates, QA, VMAF
  helper, and safe-mode LLM/VLM advisory planning.
- Soft stabilize fallback to `deshake` when vid.stab is unavailable.
