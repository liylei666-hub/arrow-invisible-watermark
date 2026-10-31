/* Node harness entry. Not for the mini program — it uses fs.
 *
 *   node js_decode/decode_cli.js --encode 13800138000
 *   node js_decode/decode_cli.js plate.raw [--debug]
 *
 * plate.raw is uint32-LE width, uint32-LE height, then width*height gray bytes.
 */
"use strict";

var fs = require("fs");
var edge = require("./arrow_edge.js");

function usage() {
  process.stderr.write(
    "usage: node js_decode/decode_cli.js --encode <11-digits>\n" +
    "       node js_decode/decode_cli.js <raw-file> [--debug]\n"
  );
}

function readRaw(path) {
  var buf = fs.readFileSync(path);
  if (buf.length < 8) {
    throw new Error("raw file shorter than header");
  }
  var width = buf.readUInt32LE(0);
  var height = buf.readUInt32LE(4);
  var need = 8 + width * height;
  if (buf.length < need) {
    throw new Error("raw file truncated: have " + buf.length + " need " + need);
  }
  return {
    width: width,
    height: height,
    gray: buf.subarray(8, need)
  };
}

function main() {
  var args = process.argv.slice(2);
  if (!args.length || args[0] === "--help" || args[0] === "-h") {
    usage();
    process.exit(args.length ? 0 : 2);
  }
  if (args[0] === "--encode") {
    if (!args[1]) {
      usage();
      process.exit(2);
    }
    process.stdout.write(edge.encodePayload(args[1]).join("") + "\n");
    return;
  }
  var debug = args.indexOf("--debug") >= 0;
  var file = args[0];
  var image = readRaw(file);
  var result = edge.decodeCleanPlate(image.gray, image.width, image.height, {
    debug: debug
  });
  process.stdout.write(JSON.stringify(result) + "\n");
}

try {
  main();
} catch (err) {
  process.stderr.write(String(err && err.stack ? err.stack : err) + "\n");
  process.exit(1);
}
