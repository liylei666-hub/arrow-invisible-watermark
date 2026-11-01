/* ARROW edge-carrier decode, clean-plate path.
 *
 * WeChat mini program: ES5, CommonJS (`require` / `module.exports`), no DOM,
 * no Node builtins. Copy this file together with `layout.js`.
 *
 * What this is
 * ------------
 * The production edge decoder measures half-width along 140 precomputed cells,
 * differentially demodulates 70 bits, soft-decodes Hamming(7,4), and accepts
 * only when the payload is 11 digits, margin >= 1.8, and registration >= 0.85.
 * This file is that measurement and that gate.
 *
 * What this is not
 * ----------------
 * There is no SIFT, no homography, and no ECC. The image must already be the
 * 2200×600 grayscale canvas (the same pixels `gen_plate.py` writes). Phone
 * photos still need the Python registration stack; a translated or perspective
 * plate will not be pulled back onto the cells.
 *
 * Registration stand-in
 * ---------------------
 * `cc` is the normalized correlation of a 4×4 box-downsample of the canvas
 * against the unwatermarked template (the same pixels as OpenCV INTER_AREA
 * at 1/4 scale). It is not `cv2.findTransformECC`. On a clean watermarked
 * plate both numbers sit near 0.997 and clear 0.85; do not treat them as
 * interchangeable on damaged photos.
 *
 * `logo_scale` is the design reference-quad width divided by 2200, i.e. the
 * value Python reports when the image→plate homography is the identity.
 * ECC on a clean plate moves it by a few ten-thousandths.
 */
"use strict";

var LAYOUT = require("./layout.js");

var CANVAS_W = LAYOUT.canvasW;
var CANVAS_H = LAYOUT.canvasH;
var QUARTER_W = LAYOUT.quarterW;
var QUARTER_H = LAYOUT.quarterH;
var N_BITS = LAYOUT.nBits;
var N_CELLS = LAYOUT.nCells;
var MARGIN_THRESHOLD = 1.8;
var CC_THRESHOLD = 0.85;
var CELLS = LAYOUT.cells;

var REASON_TEXT = {
  low_margin: "码字裕度不足",
  low_cc: "配准 cc 不足",
  invalid_payload: "码字越界",
  unreadable: "读不到图像",
  sync_failed: "同步失败",
  decode_failed: "解码失败",
  rectify_failed: "矫正失败",
  accepted: "接受"
};

function reasonText(reason) {
  if (!reason || reason === "accepted") {
    return REASON_TEXT.accepted;
  }
  var parts = String(reason).split(",");
  var out = [];
  var i;
  for (i = 0; i < parts.length; i++) {
    if (!parts[i]) {
      continue;
    }
    out.push(REASON_TEXT[parts[i]] || parts[i]);
  }
  return out.length ? out.join("，") : "拒绝";
}

/* Round half to even, matching numpy.rint / OpenCV's integer resize. */
function rint(x) {
  var neg = x < 0;
  var ax = neg ? -x : x;
  var fl = Math.floor(ax);
  var frac = ax - fl;
  var rounded;
  if (frac > 0.5) {
    rounded = fl + 1;
  } else if (frac < 0.5) {
    rounded = fl;
  } else if ((fl & 1) === 0) {
    rounded = fl;
  } else {
    rounded = fl + 1;
  }
  return neg ? -rounded : rounded;
}

function hexToBytes(hex) {
  var n = hex.length >> 1;
  var out = new Uint8Array(n);
  var i;
  for (i = 0; i < n; i++) {
    out[i] = parseInt(hex.substr(i * 2, 2), 16);
  }
  return out;
}

function expandRle(bytes) {
  var total = 0;
  var i;
  for (i = 0; i < bytes.length; i += 2) {
    total += bytes[i];
  }
  var out = new Uint8Array(total);
  var p = 0;
  var k;
  for (i = 0; i < bytes.length; i += 2) {
    var count = bytes[i];
    var value = bytes[i + 1];
    for (k = 0; k < count; k++) {
      out[p++] = value;
    }
  }
  return out;
}

var TEMPLATE_Q = expandRle(hexToBytes(LAYOUT.templateRleHex));
if (TEMPLATE_Q.length !== QUARTER_W * QUARTER_H) {
  throw new Error("quarter template length mismatch");
}

function hamming74Encode4(d1, d2, d3, d4) {
  var p1 = (d1 ^ d2 ^ d4) & 1;
  var p2 = (d1 ^ d3 ^ d4) & 1;
  var p4 = (d2 ^ d3 ^ d4) & 1;
  return [p1, p2, d1 & 1, p4, d2 & 1, d3 & 1, d4 & 1];
}

var CODEBOOK = [];
var CODEBOOK_SGN = [];
(function buildCodebook() {
  var v;
  var j;
  for (v = 0; v < 16; v++) {
    var bits = hamming74Encode4((v >> 3) & 1, (v >> 2) & 1, (v >> 1) & 1, v & 1);
    CODEBOOK.push(bits);
    var sgn = [];
    for (j = 0; j < 7; j++) {
      sgn.push(bits[j] ? 1 : -1);
    }
    CODEBOOK_SGN.push(sgn);
  }
})();

function payloadToBits(digits) {
  if (typeof digits !== "string" || !/^[0-9]{11}$/.test(digits)) {
    throw new Error("payload must be 11 digits");
  }
  var v = +digits;
  if (v >= 137438953472) {
    throw new Error("payload exceeds 37 bits");
  }
  var bits = [];
  var x = v;
  var i;
  for (i = 0; i < 37; i++) {
    bits.push(x % 2);
    x = Math.floor(x / 2);
  }
  bits.reverse();
  return bits;
}

function encodePayload(digits) {
  var b37 = payloadToBits(digits);
  var data = [0, 0, 0].concat(b37);
  var out = [];
  var i;
  for (i = 0; i < data.length; i += 4) {
    var cw = hamming74Encode4(data[i], data[i + 1], data[i + 2], data[i + 3]);
    out.push(cw[0], cw[1], cw[2], cw[3], cw[4], cw[5], cw[6]);
  }
  return out;
}

function bitsToPayload(bits37) {
  var v = 0;
  var i;
  for (i = 0; i < bits37.length; i++) {
    v = v * 2 + (bits37[i] & 1);
  }
  if (v >= 100000000000) {
    return null;
  }
  var s = String(v);
  while (s.length < 11) {
    s = "0" + s;
  }
  return s;
}

function decodePayloadSoft(soft) {
  var data = [];
  var nChanged = 0;
  var groups = (soft.length / 7) | 0;
  var g;
  var c;
  var j;
  for (g = 0; g < groups; g++) {
    var best = 0;
    var bestScore = -Infinity;
    var base = g * 7;
    for (c = 0; c < 16; c++) {
      var score = 0;
      var sgn = CODEBOOK_SGN[c];
      for (j = 0; j < 7; j++) {
        score += sgn[j] * soft[base + j];
      }
      if (score > bestScore) {
        bestScore = score;
        best = c;
      }
    }
    var code = CODEBOOK[best];
    data.push(code[2], code[4], code[5], code[6]);
    for (j = 0; j < 7; j++) {
      var hard = soft[base + j] > 0 ? 1 : 0;
      if (code[j] !== hard) {
        nChanged++;
      }
    }
  }
  return {
    digits: bitsToPayload(data.slice(3)),
    nChanged: nChanged
  };
}

function stdPop(values) {
  var n = values.length;
  if (!n) {
    return 0;
  }
  var mean = 0;
  var i;
  for (i = 0; i < n; i++) {
    mean += values[i];
  }
  mean /= n;
  var acc = 0;
  for (i = 0; i < n; i++) {
    var d = values[i] - mean;
    acc += d * d;
  }
  return Math.sqrt(acc / n);
}

function codewordMargin(soft) {
  var sd = stdPop(soft) || 1e-9;
  var groups = (soft.length / 7) | 0;
  if (!groups) {
    return 0;
  }
  var total = 0;
  var g;
  var c;
  var j;
  for (g = 0; g < groups; g++) {
    var first = -Infinity;
    var second = -Infinity;
    var base = g * 7;
    for (c = 0; c < 16; c++) {
      var score = 0;
      var sgn = CODEBOOK_SGN[c];
      for (j = 0; j < 7; j++) {
        score += sgn[j] * soft[base + j];
      }
      if (score > first) {
        second = first;
        first = score;
      } else if (score > second) {
        second = score;
      }
    }
    total += (first - second) / sd;
  }
  return total / groups;
}

function sortNum(values) {
  return values.slice().sort(function (a, b) {
    return a - b;
  });
}

function median(values) {
  if (!values.length) {
    return 0;
  }
  var s = sortNum(values);
  var n = s.length;
  var mid = n >> 1;
  if (n & 1) {
    return s[mid];
  }
  return 0.5 * (s[mid - 1] + s[mid]);
}

/* np.percentile(..., 97) linear: index = 0.97 * (n - 1). */
function percentile97(values) {
  var n = values.length;
  if (!n) {
    return 0;
  }
  var s = sortNum(values);
  var k = 0.97 * (n - 1);
  var lo = Math.floor(k);
  var hi = Math.ceil(k);
  if (lo === hi) {
    return s[lo];
  }
  return s[lo] + (s[hi] - s[lo]) * (k - lo);
}

function boxBlur3(row) {
  var n = row.length;
  var out = new Array(n);
  var i;
  for (i = 0; i < n; i++) {
    var sum = row[i];
    if (i > 0) {
      sum += row[i - 1];
    }
    if (i + 1 < n) {
      sum += row[i + 1];
    }
    out[i] = sum / 3;
  }
  return out;
}

function fracDiv(num, den) {
  var d = Math.abs(den) > 1e-3 ? den : 1e-3;
  var f = num / d;
  if (f < 0) {
    return 0;
  }
  if (f > 1) {
    return 1;
  }
  return f;
}

/**
 * Half-width + sine matched filter + paired differential features.
 * Same arithmetic as WatermarkDecoder._extract_widths_fast, without remap:
 * each row is an integer window (round-half-to-even start column).
 */
function extractFeatures(gray, width, height) {
  var profs = [];
  var allw = [];
  var ci;
  for (ci = 0; ci < CELLS.length; ci++) {
    var cell = CELLS[ci];
    var y0 = cell.y0 > 0 ? cell.y0 : 0;
    var y1 = cell.y1 < height ? cell.y1 : height;
    var half = cell.half_win | 0;
    var span = 2 * half + 2;
    var ys = [];
    var ws = [];
    var y;
    for (y = y0; y < y1; y++) {
      var cx = cell.x_mid + cell.slope * (y - cell.y_mid);
      var x0i = rint(cx - half);
      if (x0i < 0) {
        x0i = 0;
      }
      if (x0i > width - 1) {
        x0i = width - 1;
      }
      var seg = new Array(span);
      var j;
      var row = y * width;
      for (j = 0; j < span; j++) {
        var x = x0i + j;
        if (x < 0) {
          x = 0;
        } else if (x > width - 1) {
          x = width - 1;
        }
        seg[j] = gray[row + x];
      }
      var edges = seg.slice(0, 5).concat(seg.slice(span - 5));
      var bg = median(edges);
      var pv = new Array(span);
      for (j = 0; j < span; j++) {
        var ink = bg - seg[j];
        pv[j] = ink > 0 ? ink : 0;
      }
      pv = boxBlur3(pv);
      var ct = percentile97(pv);
      if (ct < 18) {
        continue;
      }
      var thr = 0.5 * ct;
      var first = -1;
      var last = -1;
      for (j = 0; j < span; j++) {
        if (pv[j] >= thr) {
          first = j;
          break;
        }
      }
      if (first < 0) {
        continue;
      }
      for (j = span - 1; j >= 0; j--) {
        if (pv[j] >= thr) {
          last = j;
          break;
        }
      }
      var left;
      var right;
      if (first > 0) {
        var i0 = first;
        if (i0 < 1) {
          i0 = 1;
        }
        if (i0 > span - 1) {
          i0 = span - 1;
        }
        left = (i0 - 1) + fracDiv(thr - pv[i0 - 1], pv[i0] - pv[i0 - 1]);
      } else {
        left = 0;
      }
      if (last < span - 1) {
        var j0 = last;
        if (j0 < 0) {
          j0 = 0;
        }
        if (j0 > span - 2) {
          j0 = span - 2;
        }
        right = j0 + fracDiv(pv[j0] - thr, pv[j0] - pv[j0 + 1]);
      } else {
        right = span - 1;
      }
      ys.push(y);
      ws.push(right - left + 1);
    }
    profs.push({ys: ys, ws: ws});
    var t;
    for (t = 0; t < ws.length; t++) {
      allw.push(ws[t]);
    }
  }

  var wmean = allw.length ? median(allw) : 1;
  var feat = new Array(CELLS.length);
  var i;
  for (i = 0; i < N_BITS; i++) {
    var ka = 2 * i;
    var kb = ka + 1;
    var pa = profs[ka];
    var pb = profs[kb];
    if (pa.ws.length < 4 || pb.ws.length < 4) {
      feat[ka] = pa.ws.length ? mean(pa.ws) : 0;
      feat[kb] = pb.ws.length ? mean(pb.ws) : 0;
      continue;
    }
    var base = 0.5 * (mean(pa.ws) + mean(pb.ws));
    feat[ka] = project(pa, CELLS[ka], base);
    feat[kb] = project(pb, CELLS[kb], base);
  }
  return {feat: feat, wmean: wmean};
}

function mean(values) {
  var s = 0;
  var i;
  for (i = 0; i < values.length; i++) {
    s += values[i];
  }
  return s / values.length;
}

function project(prof, cell, base) {
  var height = cell.y1 - cell.y0;
  var acc = 0;
  var sw = 0;
  var i;
  for (i = 0; i < prof.ws.length; i++) {
    var t = (prof.ys[i] - cell.y0 + 0.5) / height;
    var wsin = Math.sin(Math.PI * t);
    acc += (prof.ws[i] - base) * wsin;
    sw += wsin;
  }
  if (sw < 1e-6) {
    sw = 1e-6;
  }
  return acc / sw;
}

function decideBits(feat, wmean) {
  var wm = wmean > 1 ? wmean : 1;
  var bits = [];
  var conf = [];
  var i;
  for (i = 0; i < N_BITS; i++) {
    var d = feat[2 * i] - feat[2 * i + 1];
    bits.push(d > 0 ? 1 : 0);
    conf.push(d / (2 * wm));
  }
  return {bits: bits, conf: conf};
}

function quarterU8(gray, width) {
  var out = new Uint8Array(QUARTER_W * QUARTER_H);
  var y;
  var x;
  for (y = 0; y < QUARTER_H; y++) {
    for (x = 0; x < QUARTER_W; x++) {
      var sum = 0;
      var dy;
      var y0 = y * 4;
      var x0 = x * 4;
      for (dy = 0; dy < 4; dy++) {
        var row = (y0 + dy) * width + x0;
        sum += gray[row] + gray[row + 1] + gray[row + 2] + gray[row + 3];
      }
      var q = rint(sum / 16);
      if (q < 0) {
        q = 0;
      } else if (q > 255) {
        q = 255;
      }
      out[y * QUARTER_W + x] = q;
    }
  }
  return out;
}

function ncc(a, b) {
  var n = a.length;
  var ma = 0;
  var mb = 0;
  var i;
  for (i = 0; i < n; i++) {
    ma += a[i];
    mb += b[i];
  }
  ma /= n;
  mb /= n;
  var dot = 0;
  var na = 0;
  var nb = 0;
  for (i = 0; i < n; i++) {
    var da = a[i] - ma;
    var db = b[i] - mb;
    dot += da * db;
    na += da * da;
    nb += db * db;
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-12);
}

function registrationCc(gray, width, height) {
  if (width !== CANVAS_W || height !== CANVAS_H) {
    return 0;
  }
  return ncc(quarterU8(gray, width), TEMPLATE_Q);
}

function finite(value, fallback) {
  var n = +value;
  if (n !== n || n === Infinity || n === -Infinity) {
    return fallback;
  }
  return n;
}

function legalPayload(digits) {
  return typeof digits === "string" && /^[0-9]{11}$/.test(digits);
}

/**
 * Same accept rule as arrow_decode.apply_gate.
 * On reject, `digits` is null. `candidate_digits` is diagnostic only.
 */
function applyGate(raw, marginThreshold, ccThreshold) {
  var ok = !!raw.ok;
  var candidate = raw.digits == null ? null : String(raw.digits);
  if (candidate === "") {
    candidate = null;
  }
  var legal = legalPayload(candidate);
  var payloadFlag = raw.payload_valid;
  if (payloadFlag == null) {
    payloadFlag = legal;
  }
  var payloadValid = !!payloadFlag && legal;
  var margin = finite(raw.margin, 0);
  var cc = finite(raw.cc, 0);
  var meanConf = finite(raw.mean_conf, 0);
  var logo = raw.logo_scale;
  if (logo != null) {
    logo = finite(logo, null);
    if (logo !== logo) {
      logo = null;
    }
  }
  var reasons = [];
  if (!ok) {
    reasons.push(raw.reason ? String(raw.reason) : "decode_failed");
  } else {
    if (!payloadValid) {
      reasons.push("invalid_payload");
    }
    if (ccThreshold != null && cc < ccThreshold) {
      reasons.push("low_cc");
    }
    if (margin < marginThreshold) {
      reasons.push("low_margin");
    }
  }
  var accepted = reasons.length === 0;
  return {
    accepted: accepted,
    digits: accepted ? candidate : null,
    candidate_digits: candidate,
    carrier: "edge",
    margin: margin,
    cc: cc,
    logo_scale: logo == null ? null : logo,
    mean_conf: meanConf,
    payload_valid: payloadValid,
    ok: ok,
    reason: accepted ? "" : reasons.join(","),
    reason_text: reasonText(accepted ? "accepted" : reasons.join(",")),
    margin_threshold: marginThreshold,
    cc_threshold: ccThreshold,
    cc_method: "quarter_res_ncc",
    path: "clean-plate"
  };
}

function fail(reason, detail, marginThreshold, ccThreshold) {
  var raw = {
    ok: false,
    reason: reason,
    digits: null,
    margin: 0,
    cc: 0,
    mean_conf: 0,
    logo_scale: null,
    payload_valid: false
  };
  var gated = applyGate(raw, marginThreshold, ccThreshold);
  if (detail) {
    gated.detail = detail;
  }
  return gated;
}

/**
 * Decode one grayscale image.
 *
 * gray: Uint8Array or array-like, row-major, length width*height, levels 0–255.
 * Only width=2200 and height=600 is a successful sync. Anything else is
 * sync_failed — this path does not search for the logo.
 *
 * options.marginThreshold (default 1.8)
 * options.ccThreshold (default 0.85; pass null to skip the registration gate)
 * options.debug: include widths, conf, bits, wmean
 */
function decodeCleanPlate(gray, width, height, options) {
  options = options || {};
  var marginThreshold = options.marginThreshold != null ? +options.marginThreshold : MARGIN_THRESHOLD;
  var ccThreshold = options.ccThreshold === null
    ? null
    : (options.ccThreshold != null ? +options.ccThreshold : CC_THRESHOLD);
  width = width | 0;
  height = height | 0;
  if (!gray || width <= 0 || height <= 0 || gray.length !== width * height) {
    var detail = !gray ? "empty array" : "length " + (gray ? gray.length : 0) +
      " != " + width + "x" + height;
    return fail("unreadable", detail, marginThreshold, ccThreshold);
  }
  if (width !== CANVAS_W || height !== CANVAS_H) {
    return fail(
      "sync_failed",
      "clean-plate path needs " + CANVAS_W + "x" + CANVAS_H +
        ", got " + width + "x" + height,
      marginThreshold,
      ccThreshold
    );
  }

  var extracted = extractFeatures(gray, width, height);
  var decided = decideBits(extracted.feat, extracted.wmean);
  var soft = decodePayloadSoft(decided.conf);
  var meanConf = 0;
  var i;
  for (i = 0; i < decided.conf.length; i++) {
    var c = decided.conf[i];
    meanConf += c < 0 ? -c : c;
  }
  meanConf = decided.conf.length ? meanConf / decided.conf.length : 0;
  var raw = {
    ok: true,
    digits: soft.digits,
    payload_valid: soft.digits != null,
    margin: codewordMargin(decided.conf),
    cc: registrationCc(gray, width, height),
    logo_scale: LAYOUT.logoScale,
    mean_conf: meanConf
  };
  var gated = applyGate(raw, marginThreshold, ccThreshold);
  if (options.debug) {
    gated.widths = extracted.feat;
    gated.conf = decided.conf;
    gated.bits = decided.bits;
    gated.wmean = extracted.wmean;
    gated.n_corrected = soft.nChanged;
  }
  return gated;
}

module.exports = {
  decodeCleanPlate: decodeCleanPlate,
  encodePayload: encodePayload,
  decodePayloadSoft: decodePayloadSoft,
  codewordMargin: codewordMargin,
  reasonText: reasonText,
  MARGIN_THRESHOLD: MARGIN_THRESHOLD,
  CC_THRESHOLD: CC_THRESHOLD,
  CANVAS_W: CANVAS_W,
  CANVAS_H: CANVAS_H,
  cellCount: N_CELLS,
  registration: "quarter-ncc",
  path: "clean-plate"
};
