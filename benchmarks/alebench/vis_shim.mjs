// Docker-free official scorer: drives the problem's own WASM visualizer (gen + vis) from the ALE-Bench zip.
// Usage:
//   node vis_shim.mjs <problem_dir> gen <seed>                         -> the input on stdout
//   node vis_shim.mjs <problem_dir> score <input_file> <output_file>   -> {"score":..,"err":..,"turns":..}
//   node vis_shim.mjs <problem_dir> score-many <pairs.json>            -> one JSON line per [input_file, output_file]
// The visualizer is the contest's own scoring code compiled to WASM, so scores match AtCoder's tester.
import fs from "node:fs";
import path from "node:path";
import { pathToFileURL } from "node:url";

const [, , pdir, cmd, a, b] = process.argv;
const vdir = path.join(pdir, "visualization_server");
const candidates = fs.readdirSync(vdir).filter(
  (f) => f.endsWith(".js") && fs.existsSync(path.join(vdir, f.replace(/\.js$/, "_bg.wasm"))));
// Prefer the contest build over the XXXXXXXX placeholder build when both ship.
const js = candidates.find((f) => !f.startsWith("XXXXXXXX")) || candidates[0];
const mod = await import(pathToFileURL(path.join(vdir, js)).href);
const wbytes = fs.readFileSync(path.join(vdir, js.replace(/\.js$/, "_bg.wasm")));
try { mod.initSync({ module: wbytes }); } catch (e) { mod.initSync(wbytes); }

function scoreOne(input, output) {
  try {
    const t = mod.get_max_turn ? mod.get_max_turn(input, output) : 0;
    const r = mod.vis(input, output, t, mod.VisOption ? new mod.VisOption() : "");
    // wasm-bindgen exposes the message as `error` (older builds as `err`); a non-empty one means an illegal output.
    const err = String(r.error ?? r.err ?? "");
    return { score: err ? 0 : Number(r.score), err, turns: t };
  } catch (e) {
    return { score: 0, err: `visualizer error: ${String(e).slice(0, 200)}`, turns: 0 };
  }
}

if (cmd === "gen") {
  process.stdout.write(mod.gen(String(a), mod.GenOption ? new mod.GenOption() : ""));
} else if (cmd === "score") {
  console.log(JSON.stringify(scoreOne(fs.readFileSync(a, "utf8"), fs.readFileSync(b, "utf8"))));
} else if (cmd === "score-many") {
  for (const [inp, out] of JSON.parse(fs.readFileSync(a, "utf8"))) {
    console.log(JSON.stringify(scoreOne(fs.readFileSync(inp, "utf8"), fs.readFileSync(out, "utf8"))));
  }
} else {
  console.error(`unknown command ${cmd}`);
  process.exit(2);
}
