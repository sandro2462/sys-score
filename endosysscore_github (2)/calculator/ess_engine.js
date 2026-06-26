/* EndoSysScore — pure-JavaScript inference engine.
   No WebAssembly, no onnxruntime, no CDN. Runs the real deposited model entirely in plain JS.
   Pipeline: 124-feature vector -> xgb_shallow, xgb_deep, logreg -> LightGBM meta -> isotonic -> beta. */
(function (global) {
  "use strict";
  function b(x){return x?1:0;}
  function sig(z){return 1/(1+Math.exp(-z));}
  function buildRow(inp){
    var age=+(inp.age!=null?inp.age:65), lvef=+(inp.lvef!=null?inp.lvef:55);
    var pve=b(inp.pve), nve=1-pve;
    return {ETA:age, female:b(inp.female), eta_ge80:b(age>=80), eta_class:Math.floor(age/10),
      eta_lt60:b(age<60), eta_60_69:b(age>=60&&age<70), eta_70_79:b(age>=70&&age<80),
      NVE:nve, PVE:pve, lvef:lvef, lvef_class_low:b(lvef<40), lvef_class_mid:b(lvef>=40&&lvef<50),
      PAPSgt50:b(inp.paps), SCOMPENSO:b(inp.hf),
      AO:b(inp.aortic), AO_final:b(inp.aortic), M:b(inp.mitral), M_final:b(inp.mitral),
      IM:b(inp.mitral), isolated_IM:b(inp.isomit), multivalve:b(inp.multi), multivalve_final:b(inp.multi),
      periannular:b(inp.periann), ASCESSO:b(inp.abscess), FISTOLA:0, PSEUDOANEURISMA:0, VEGETAZIONI:1,
      BPCO:b(inp.copd), IRC_stage:(inp.ckd?3:0), DIALISI:b(inp.dialysis),
      saureus_b:b(inp.saureus), culture_neg:b(inp.cultneg), fungal_b:b(inp.fungal),
      SHOCK:b(inp.shock), INTUBPRE:b(inp.intub), IABP_PRE:b(inp.iabp), ENDOATTIVA:b(inp.active)};
  }
  function patterns(row, lvefMed){
    var lv=(row.lvef==null||isNaN(row.lvef))?lvefMed:row.lvef, p={};
    p.in_pattern_A=b(row.multivalve_final===1&&row.saureus_b===1&&(row.IRC_stage>=2||row.DIALISI===1||row.PAPSgt50===1));
    p.in_pattern_B=b(row.ETA>=80&&row.NVE===1&&(row.BPCO===1||lv<50));
    p.in_pattern_C=b((row.ASCESSO===1||row.periannular===1)&&row.isolated_IM===1&&row.ETA>=65);
    p.in_pattern_D=b(row.PVE===1&&row.saureus_b===1&&row.IM===1&&row.AO_final===1);
    p.in_pattern_E=b(row.NVE===1&&row.ETA>=80&&lv<50);
    p.in_pattern_F=b(row.ASCESSO===1&&row.isolated_IM===1&&row.ETA>=65);
    p.n_patterns_satisfied=p.in_pattern_A+p.in_pattern_B+p.in_pattern_C+p.in_pattern_D+p.in_pattern_E+p.in_pattern_F;
    return p;
  }
  function buildVector(inp, featCols, lvefMed){
    var row=buildRow(inp), pat=patterns(row,lvefMed);
    for(var k in pat) row[k]=pat[k];
    var v=new Float32Array(featCols.length);
    for(var i=0;i<featCols.length;i++){var c=featCols[i]; v[i]=(row[c]!=null&&!isNaN(row[c]))?row[c]:0;}
    return {vec:v, patterns:pat};
  }
  function xgbPredict(mdl, x){
    var tot=mdl.intercept, trees=mdl.trees;
    for(var t=0;t<trees.length;t++){
      var n=trees[t];
      while(n.leaf===undefined){
        var f=+n.split.slice(1), nxt=(x[f] < n.split_condition)?n.yes:n.no;
        if(isNaN(x[f])) nxt=n.missing;
        var ch=n.children;
        for(var j=0;j<ch.length;j++){ if(ch[j].nodeid===nxt){ n=ch[j]; break; } }
      }
      tot+=n.leaf;
    }
    return sig(tot);
  }
  function lgbRaw(model, x){
    var raw=0, ti=model.tree_info;
    for(var t=0;t<ti.length;t++){
      var n=ti[t].tree_structure;
      while(n.leaf_value===undefined){
        var f=n.split_feature, thr=n.threshold;
        var goLeft=(n.decision_type==='<=')?(x[f]<=thr):(x[f]<thr);
        if(isNaN(x[f])) goLeft=(n.default_left!==false);
        n=goLeft?n.left_child:n.right_child;
      }
      raw+=n.leaf_value;
    }
    return raw;
  }
  function logregPredict(lg, x){
    var z=lg.intercept, c=lg.coef, m=lg.mean, s=lg.scale;
    for(var j=0;j<c.length;j++) z+=c[j]*((x[j]-m[j])/s[j]);
    return sig(z);
  }
  function isotonic(p, xs, ys){
    if(p<=xs[0])return ys[0]; if(p>=xs[xs.length-1])return ys[ys.length-1];
    var lo=0,hi=xs.length-1; while(hi-lo>1){var mid=(lo+hi)>>1; if(xs[mid]<=p)lo=mid; else hi=mid;}
    var t=(p-xs[lo])/(xs[hi]-xs[lo]); return ys[lo]+t*(ys[hi]-ys[lo]);
  }
  function beta(p, coef, intercept){
    // betacal clips input to [eps,1-eps] (float64 epsilon), applies the logistic map, then output clipped to 1e-6
    var EPS=Number.EPSILON;                       // 2.220446049250313e-16
    var pc=Math.min(Math.max(p,EPS),1-EPS);
    var z=coef[0]*Math.log(pc)+coef[1]*(-Math.log(1-pc))+intercept;
    return Math.min(Math.max(sig(z),1e-6),1-1e-6);
  }
  function predict(MODEL, inp){
    var cal=MODEL.calib;
    var bv=buildVector(inp, cal.feat_cols, cal.lvef_med), v=bv.vec;
    var fr=Math.fround;
    var bs=fr(xgbPredict(MODEL.xgb_shallow, v)), bd=fr(xgbPredict(MODEL.xgb_deep, v)), bl=fr(logregPredict(MODEL.logreg, v));
    var mf=[bs,bd,bl];
    cal.pat_feat_idx.forEach(function(i){mf.push(v[i]);});
    cal.clinical_idx.forEach(function(i){mf.push(v[i]);});
    var mp=sig(lgbRaw(MODEL.lgbm, mf));
    var iso=isotonic(mp, cal.iso_x, cal.iso_y);
    var fin=beta(iso, cal.beta_coef, cal.beta_intercept);
    return {percent: fin*100, prob: fin, patterns: bv.patterns};
  }
  var api={buildVector:buildVector, xgbPredict:xgbPredict, lgbRaw:lgbRaw, logregPredict:logregPredict,
           isotonic:isotonic, beta:beta, predict:predict};
  if(typeof module!=="undefined"&&module.exports) module.exports=api;
  global.ESSEngine=api;
})(typeof window!=="undefined"?window:globalThis);
