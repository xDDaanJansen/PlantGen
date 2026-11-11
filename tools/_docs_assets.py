"""CSS / JS / HTML shell for tools/build_docs.py, kept apart so the build logic
stays readable. The palette and type system are the project's own, extended
with a ToC sidebar and client-side search.

TEMPLATE is a str.format template, so it must contain no literal braces except
its named fields ({css} {js} {toc} {pages} {reference} {n_fields} {n_kinds}).
CSS and JS are inserted as VALUES, so their braces are safe."""

# ruff: noqa: E501  -- CSS/JS/HTML asset blobs; long lines are inherent.

CSS = r"""
:root{
  --paper:#EFEEE3; --paper-raised:#F8F7EF; --paper-inset:#E6E4D5;
  --ink:#1E2B22; --ink-soft:#4B5647; --ink-faint:#7C8574;
  --rule:#D6D2BF; --rule-soft:#E2DFCF;
  --keep:#3B5D42; --keep-soft:#E1E8DC;
  --new:#4F6D82; --new-soft:#E1E7EC; --amber:#8C6A2E; --link:#3E6E8E;
  --flash:#FBF3D6;
  --amber-soft:#F1E8D6; --violet:#6A5A8E; --violet-soft:#E8E4F0; --rust:#9A4E38; --rust-soft:#F3E2DB;
  --font-display:Georgia,'Iowan Old Style','Palatino Linotype','Times New Roman',serif;
  --font-body:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
  --font-mono:ui-monospace,'Cascadia Code','SF Mono',Consolas,'Liberation Mono',monospace;
}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){
  --paper:#14170F; --paper-raised:#1B1F16; --paper-inset:#10130C;
  --ink:#E9E7D9; --ink-soft:#B7B8A9; --ink-faint:#82877A;
  --rule:#333A2C; --rule-soft:#262C21;
  --keep:#83AF8E; --keep-soft:#202B22;
  --new:#8FB0C4; --new-soft:#1C262C; --amber:#CBA35C; --link:#8FC0DD;
  --flash:#2E2A16;
  --amber-soft:#2B2416; --violet:#B6A6DA; --violet-soft:#24212E; --rust:#DD9783; --rust-soft:#2E1F1A;
}}
:root[data-theme="dark"]{
  --paper:#14170F; --paper-raised:#1B1F16; --paper-inset:#10130C;
  --ink:#E9E7D9; --ink-soft:#B7B8A9; --ink-faint:#82877A;
  --rule:#333A2C; --rule-soft:#262C21;
  --keep:#83AF8E; --keep-soft:#202B22;
  --new:#8FB0C4; --new-soft:#1C262C; --amber:#CBA35C; --link:#8FC0DD; --flash:#2E2A16;
  --amber-soft:#2B2416; --violet:#B6A6DA; --violet-soft:#24212E; --rust:#DD9783; --rust-soft:#2E1F1A;
}
*{box-sizing:border-box}
@media (prefers-reduced-motion:reduce){*{animation-duration:.001ms!important;transition-duration:.001ms!important}}
html,body{margin:0;padding:0}
body{background:var(--paper);color:var(--ink);font-family:var(--font-body);font-size:15px;line-height:1.56}
a{color:var(--link);text-underline-offset:2px}
a:focus-visible{outline:2px solid var(--new);outline-offset:2px}
code,.mono{font-family:var(--font-mono);font-size:.88em}
h1,h2,h3,h4{font-family:var(--font-display);font-weight:600;text-wrap:balance;color:var(--ink);margin:0}

.masthead{position:sticky;top:0;z-index:30;border-bottom:1px solid var(--rule);background:var(--paper)}
.masthead-inner{max-width:1220px;margin:0 auto;padding:10px 24px;display:flex;align-items:center;justify-content:space-between;gap:16px}
.masthead-title{font-family:var(--font-display);font-size:16px;font-weight:600;letter-spacing:.02em}
.masthead-tools{display:flex;align-items:center;gap:10px}
.search{font-family:var(--font-body);font-size:13.5px;padding:7px 11px;width:min(46vw,340px);
  background:var(--paper-raised);border:1px solid var(--rule);border-radius:6px;color:var(--ink)}
.search:focus{outline:none;border-color:var(--new)}
.theme-btn{font-size:15px;background:var(--paper-raised);border:1px solid var(--rule);border-radius:6px;
  color:var(--ink-soft);width:32px;height:32px;cursor:pointer}
.results{position:absolute;top:52px;right:24px;width:min(90vw,420px);max-height:64vh;overflow-y:auto;
  background:var(--paper-raised);border:1px solid var(--rule);border-radius:8px;box-shadow:0 8px 30px rgba(0,0,0,.18);z-index:40}
.results a{display:block;padding:9px 13px;border-bottom:1px solid var(--rule-soft);text-decoration:none;color:var(--ink)}
.results a:last-child{border-bottom:none}
.results a.sel,.results a:hover{background:var(--new-soft)}
.results .r-label{font-weight:600;font-size:13.5px}
.results .r-sub{font-family:var(--font-mono);font-size:11px;color:var(--ink-faint)}
.results .r-kind{float:right;font-family:var(--font-mono);font-size:10px;text-transform:uppercase;color:var(--keep)}
.results .r-empty{padding:12px 13px;color:var(--ink-faint);font-size:13px}

.layout{max-width:1220px;margin:0 auto;display:grid;grid-template-columns:246px minmax(0,1fr);gap:0}
.toc{position:sticky;top:53px;align-self:start;height:calc(100vh - 53px);overflow-y:auto;
  border-right:1px solid var(--rule);padding:20px 14px 60px}
.toc-inner{display:flex;flex-direction:column;gap:1px}
.toc a{text-decoration:none;color:var(--ink-soft);border-radius:5px;display:block}
.toc a:hover{background:var(--paper-inset)}
.toc-1{font-family:var(--font-mono);font-size:10.5px;text-transform:uppercase;letter-spacing:.06em;
  color:var(--ink-faint);padding:14px 8px 4px;font-weight:600}
.toc-2{font-size:13px;padding:4px 8px 4px 12px}
.toc-2.active{background:var(--new-soft);color:var(--ink);font-weight:600}
.toc-3{font-size:12px;padding:2px 8px 2px 26px;color:var(--ink-faint)}
.toc-3:hover{color:var(--ink-soft)}

.wrap{padding:0 40px 120px;min-width:0}
.intro{padding:44px 0 26px;border-bottom:1px solid var(--rule)}
.eyebrow{font-family:var(--font-mono);font-size:11px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--keep);margin-bottom:12px;display:flex;align-items:center;gap:10px}
.eyebrow::before{content:"";width:22px;height:1px;background:var(--keep)}
h1.title{font-size:31px;line-height:1.15;max-width:20ch}
.standfirst{margin-top:14px;font-size:15.5px;color:var(--ink-soft);max-width:66ch}

section{margin-top:44px;scroll-margin-top:64px}
.sec-head{display:flex;align-items:baseline;gap:12px;margin-bottom:6px;flex-wrap:wrap;
  padding-bottom:6px;border-bottom:1px solid var(--rule-soft)}
.sec-head h2{font-size:22px}
.sec-kicker{font-family:var(--font-mono);font-size:10.5px;text-transform:uppercase;letter-spacing:.05em;color:var(--ink-faint)}
.sec-intro{font-size:14px;color:var(--ink-soft);max-width:70ch;margin:10px 0 18px}
.sec-see{font-size:12.5px;color:var(--ink-faint);margin:0 0 16px}
.sec-see a{color:var(--link)}
h3.sub{font-family:var(--font-body);font-weight:700;font-size:12px;text-transform:uppercase;
  letter-spacing:.05em;color:var(--ink-soft);margin:24px 0 8px}
h4.sub{font-size:15px;margin:18px 0 6px}
.ref-group{font-family:var(--font-body);font-weight:700;font-size:15px;text-transform:uppercase;
  letter-spacing:.04em;color:var(--keep);margin:34px 0 12px;padding-bottom:6px;
  border-bottom:1px solid var(--rule)}
p{margin:0 0 13px;max-width:70ch}
ul,ol{margin:0 0 13px;padding-left:20px;max-width:70ch}
li{margin-bottom:6px}
blockquote{margin:0 0 14px;padding:9px 15px;border-left:3px solid var(--keep);background:var(--keep-soft);
  color:var(--ink-soft);font-size:14px;border-radius:0 4px 4px 0}
pre{background:var(--paper-inset);border:1px solid var(--rule);border-radius:6px;padding:12px 14px;overflow-x:auto;font-size:12.5px}
pre code{font-size:inherit}

.ref-item{padding:14px 0;border-bottom:1px solid var(--rule-soft);scroll-margin-top:64px}
.ref-item:target{background:var(--flash);margin:0 -14px;padding:14px;border-radius:6px;transition:background .2s}
.ref-item.ref-sub{margin-left:28px}
.ref-head{display:flex;align-items:baseline;gap:9px;flex-wrap:wrap}
.ref-label{font-weight:700;font-size:15px}
.ref-path{font-family:var(--font-mono);font-size:12px;color:var(--ink-faint);background:var(--paper-inset);padding:1px 6px;border-radius:4px}
.badge{font-family:var(--font-mono);font-size:9.5px;text-transform:uppercase;letter-spacing:.04em;
  padding:1px 6px;border-radius:4px;border:1px solid var(--rule)}
.badge.type,.badge.b-value{color:var(--new);background:var(--new-soft);border-color:transparent}
.badge.b-switch{color:var(--violet);background:var(--violet-soft);border-color:transparent}
.badge.b-data{color:var(--amber);background:var(--amber-soft);border-color:transparent}
.badge.b-action{color:var(--rust);background:var(--rust-soft);border-color:transparent}
/* what sits beside a control: outlined; where it shows: plain */
.badge.b-extra{color:var(--keep);background:transparent;border-color:var(--keep)}
.badge.b-debug{color:var(--rust);background:transparent;border-color:var(--rust);border-style:dashed}
.badge.b-where{color:var(--ink-faint);background:transparent;border-color:transparent;text-transform:none;letter-spacing:0;font-style:italic;padding-left:2px}
.badge[title]{cursor:help}
.badge-legend{margin:0 0 28px;padding:12px 16px;border:1px solid var(--rule);border-radius:8px;background:var(--paper-raised);display:flex;flex-direction:column;gap:7px;max-width:76ch}
.legend-row{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px}
.legend-name{font-size:12.5px;color:var(--ink-soft);min-width:128px}
.badge-legend p{margin:2px 0 0;font-size:12px;color:var(--ink-faint)}
.badge.species{color:var(--amber);background:transparent;border-color:var(--amber)}
.ref-meta{font-family:var(--font-mono);font-size:12px;color:var(--ink-soft);margin:6px 0 2px}
.ref-meta code{font-size:11.5px}
.ref-desc{font-size:14px;color:var(--ink);margin:7px 0 0;max-width:74ch}
.endpoints{list-style:none;margin:9px 0 0;padding:0;max-width:76ch;display:flex;flex-direction:column;gap:3px}
.endpoints li{display:flex;gap:10px;margin:0;font-size:13.5px}
.ep-at{font-family:var(--font-mono);font-size:12px;color:var(--keep);min-width:52px;text-align:right;flex-shrink:0}
.ep-look{color:var(--ink-soft)}
.see-also{font-size:12.5px;color:var(--ink-faint);margin-top:8px}
.see-also a{font-family:var(--font-mono);font-size:11.5px}

.table-scroll{overflow-x:auto;border:1px solid var(--rule);border-radius:6px;margin:4px 0 8px}
table{width:100%;border-collapse:collapse;font-size:13px}
table.wide{min-width:640px}
.table-scroll.md{display:inline-block;max-width:100%;margin:0 0 14px}
table.md{width:auto}
thead th{text-align:left;font-family:var(--font-mono);font-weight:500;font-size:10px;text-transform:uppercase;
  letter-spacing:.05em;color:var(--ink-faint);padding:9px 12px;background:var(--paper-raised);border-bottom:1px solid var(--rule)}
tbody td{padding:8px 12px;border-bottom:1px solid var(--rule-soft);vertical-align:top}
tbody tr:last-child td{border-bottom:none}
td.yes{color:var(--keep);font-weight:600}
td.no{color:var(--ink-faint)}
.shape-list{columns:2;max-width:60ch}
.shape-list li{break-inside:avoid}

.fig{margin:16px 0;padding:14px;background:var(--paper-raised);border:1px solid var(--rule);border-radius:8px;text-align:center}
.fig svg{max-width:100%;height:auto}
.fig figcaption{margin-top:8px;font-size:12.5px;color:var(--ink-faint)}
.ui-icon{display:inline-flex;vertical-align:-4px;margin:0 3px}
.ui-icon svg{width:17px;height:17px}

.kind-card{display:flex;gap:16px;padding:15px 0;border-bottom:1px solid var(--rule-soft);scroll-margin-top:64px}
.kind-card:last-child{border-bottom:none}
.kind-fig{flex:none;width:74px}
.kind-fig svg{width:74px;height:auto;display:block}
.kind-body{min-width:0}
.kind-name{font-weight:700;font-size:15px}
.kind-name code{font-family:var(--font-mono);font-size:12px;color:var(--ink-faint);font-weight:400;margin-left:8px}
.kind-caps{font-family:var(--font-mono);font-size:11px;color:var(--ink-soft);margin:4px 0 7px}
.kind-caps .no{color:var(--ink-faint)}
.kind-desc{font-size:14px;color:var(--ink);margin:0;max-width:64ch}
.kind-fig svg path,.kind-fig svg line{stroke:var(--ink-soft);stroke-width:2;fill:none;stroke-linecap:round;stroke-linejoin:round}
.kind-fig svg circle{fill:var(--keep)}
.kind-fig svg path.blade{fill:var(--keep);stroke:none}
.value-cards{margin:8px 0 2px}
.ui-icon{color:var(--ink-soft)}
.shape-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(94px,1fr));gap:12px;max-width:660px}
.shape-cell{text-align:center;padding:9px 8px;background:var(--paper-raised);border:1px solid var(--rule);border-radius:6px}
.curve-svg{width:100%;height:auto;display:block}
.curve-svg polyline,.curve-svg path{fill:none;stroke:var(--keep);stroke-width:2.5;stroke-linecap:round;stroke-linejoin:round}
.shape-key code{font-family:var(--font-mono);font-size:11px;color:var(--ink)}
.shape-label{font-size:11px;color:var(--ink-faint);margin-top:1px}
.value-list{margin:10px 0 2px;display:flex;flex-direction:column;gap:10px}
.value-item{display:flex;gap:12px;align-items:flex-start}
.value-fig{flex:none;width:46px}
.value-fig svg{width:46px;height:auto;display:block}
.value-fig svg path,.value-fig svg line{stroke:var(--ink-soft);stroke-width:2;fill:none;stroke-linecap:round;stroke-linejoin:round}
.value-fig svg circle{fill:var(--keep)}
.value-fig svg circle.bud{fill:none;stroke:var(--keep);stroke-width:1.6}
.value-fig svg circle.ring{fill:none;stroke:var(--keep);stroke-width:1.4;stroke-dasharray:3 2.5}
.value-fig svg circle.aim{fill:none;stroke:var(--keep);stroke-width:1.6}
.value-fig svg .zone{fill:var(--keep-soft);stroke:var(--keep);stroke-width:1.4}
.value-fig svg path.pole{stroke:var(--keep);stroke-width:3.5}
.value-fig svg path.plane{stroke:var(--keep);stroke-width:1.6;stroke-dasharray:4 3}
.value-name{font-weight:600;font-size:13.5px}
.value-item code{font-family:var(--font-mono);font-size:11.5px;color:var(--ink-faint)}
.value-desc{font-size:13px;color:var(--ink-soft);margin:2px 0 0;max-width:66ch}
.gloss{margin:14px 0}
.gloss-item{padding:9px 0;border-bottom:1px solid var(--rule-soft);scroll-margin-top:64px}
.gloss-item:last-child{border-bottom:none}
.gloss-item:target{background:var(--flash);margin:0 -12px;padding:9px 12px;border-radius:6px}
.gloss dt{font-weight:700;font-size:14px}
.gloss dd{margin:2px 0 0;font-size:13.5px;color:var(--ink-soft);max-width:72ch}

/* the panel reference: folds and sub-panels nest like the N-panel */
.ref-node{scroll-margin-top:64px}
.ref-node.depth-2,.ref-node.depth-3,.ref-node.depth-4{margin-left:14px;padding-left:12px;
  border-left:2px solid var(--rule-soft)}
h4.ref-group,h5.ref-group,h6.ref-group{font-size:13.5px;text-transform:none;letter-spacing:0}
.ref-kind{font-weight:400;font-size:11px;color:var(--ink-soft);margin-left:8px;text-transform:none;
  letter-spacing:0}
.ref-group .badge{margin-left:6px;text-transform:none;letter-spacing:0;font-weight:400}
.ref-subhead{font-size:13px;font-weight:700;margin:14px 0 0;color:var(--ink-soft)}
.ref-note{margin:6px 0 0;padding:6px 10px;border-left:3px solid var(--rule);font-size:13.5px;
  max-width:72ch}
.shot{border:1px solid var(--rule);border-radius:6px;max-width:100%;display:block;margin:10px 0}
.shot-missing{border:1px dashed var(--rule);border-radius:6px;padding:14px;margin:10px 0;
  font-size:13px;color:var(--ink-soft)}
.render{border-radius:6px;max-width:100%;display:block;margin:10px 0}

@media (max-width:900px){
  .layout{grid-template-columns:1fr}
  .toc{position:static;height:auto;border-right:none;border-bottom:1px solid var(--rule);padding:12px 20px}
  .toc-inner{flex-flow:row wrap;gap:2px 10px}
  .toc-1{padding:6px 0 2px;width:100%}
  .wrap{padding:0 20px 80px}
  .search{width:44vw}
  .results{right:8px;left:8px;width:auto}
}
"""

JS = r"""
(function(){
  var INDEX = __INDEX__;
  // theme
  var root = document.documentElement;
  try{ var t = localStorage.getItem('pg-theme'); if(t) root.setAttribute('data-theme', t); }catch(e){}
  document.getElementById('theme').addEventListener('click', function(){
    var cur = root.getAttribute('data-theme');
    var next = cur === 'dark' ? 'light' : (cur === 'light' ? 'dark' :
      (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'light':'dark'));
    root.setAttribute('data-theme', next);
    try{ localStorage.setItem('pg-theme', next); }catch(e){}
  });
  // search
  var box = document.getElementById('search'), panel = document.getElementById('results'), sel = -1, hits = [];
  function esc(s){ return s.replace(/[&<>]/g, function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;'}[c];}); }
  function run(q){
    q = q.trim().toLowerCase();
    if(!q){ panel.hidden = true; return; }
    hits = [];
    for(var i=0;i<INDEX.length;i++){
      var e = INDEX[i];
      var label=e.label.toLowerCase(), sub=(e.sub||'').toLowerCase(),
          terms=(e.terms||'').toLowerCase(), desc=(e.desc||'').toLowerCase();
      var score;
      if(label.indexOf(q)===0) score=0;              // label starts with the query
      else if(label.indexOf(q)>=0) score=1;          // label contains it
      else if(sub.indexOf(q)>=0) score=2;            // the profile key
      else if((' '+terms).indexOf(' '+q)>=0) score=3; // an enum value starts with it
      else if(terms.indexOf(q)>=0) score=4;          // an enum value contains it
      else if(desc.indexOf(q)>=0) score=5;           // somewhere in the description
      else continue;
      hits.push([score, e]);
    }
    hits.sort(function(a,b){return a[0]-b[0];});
    hits = hits.slice(0,30).map(function(h){return h[1];});
    if(!hits.length){ panel.innerHTML = '<div class="r-empty">No match for &ldquo;'+esc(q)+'&rdquo;</div>'; panel.hidden=false; return; }
    panel.innerHTML = hits.map(function(e){
      return '<a href="#'+e.id+'" data-id="'+e.id+'"><span class="r-kind">'+e.kind+'</span>'+
        '<span class="r-label">'+esc(e.label)+'</span><br><span class="r-sub">'+esc(e.sub)+'</span></a>';
    }).join('');
    sel = -1; panel.hidden = false;
  }
  function go(id){ panel.hidden=true; box.value=''; var el=document.getElementById(id);
    if(el){ history.replaceState(null,'','#'+id); el.scrollIntoView(); if(el.classList.contains('ref-item')){ location.hash=id; } } }
  box.addEventListener('input', function(){ run(box.value); });
  box.addEventListener('keydown', function(ev){
    var links = panel.hidden?[]:panel.querySelectorAll('a');
    if(ev.key==='ArrowDown'){ ev.preventDefault(); sel=Math.min(sel+1,links.length-1); }
    else if(ev.key==='ArrowUp'){ ev.preventDefault(); sel=Math.max(sel-1,0); }
    else if(ev.key==='Enter'){ ev.preventDefault(); if(links[sel]) go(links[sel].dataset.id); else if(hits[0]) go(hits[0].id); return; }
    else if(ev.key==='Escape'){ panel.hidden=true; box.blur(); return; }
    else return;
    links.forEach(function(a,i){ a.classList.toggle('sel', i===sel); if(i===sel) a.scrollIntoView({block:'nearest'}); });
  });
  panel.addEventListener('click', function(ev){ var a=ev.target.closest('a'); if(a){ ev.preventDefault(); go(a.dataset.id); } });
  document.addEventListener('click', function(ev){ if(!panel.contains(ev.target) && ev.target!==box) panel.hidden=true; });
  document.addEventListener('keydown', function(ev){ if(ev.key==='/' && document.activeElement!==box){ ev.preventDefault(); box.focus(); } });
  // scrollspy
  var links = {}; document.querySelectorAll('.toc-2').forEach(function(a){ links[a.dataset.target]=a; });
  var obs = new IntersectionObserver(function(es){
    es.forEach(function(e){ var a=links[e.target.id]; if(!a) return;
      if(e.isIntersecting){ Object.keys(links).forEach(function(k){links[k].classList.remove('active');}); a.classList.add('active');
        // keep the active link in view inside the sidebar only: in the narrow
        // layout the contents sit at the top of the page, and scrolling the
        // link into view there dragged the whole page back up
        var box = a.closest('.toc');
        if(box && box.scrollHeight > box.clientHeight + 4){
          var r = a.getBoundingClientRect(), b = box.getBoundingClientRect();
          if(r.top < b.top || r.bottom > b.bottom){ box.scrollTop += r.top - b.top - b.height / 2; } } } });
  }, {rootMargin:'-56px 0px -70% 0px'});
  document.querySelectorAll('section[id]').forEach(function(s){ obs.observe(s); });
  // an image used on several pages is embedded once; the others copy it
  document.querySelectorAll('img[data-same]').forEach(function(img){
    var src = document.getElementById(img.dataset.same); if(src) img.src = src.src; });
})();
"""

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PlantGen documentation</title>
<style>{css}</style>
</head>
<body>
<header class="masthead">
  <div class="masthead-inner">
    <span class="masthead-title">PlantGen</span>
    <div class="masthead-tools">
      <input id="search" class="search" type="search" placeholder="Search {n_fields} controls  ( / )" autocomplete="off" spellcheck="false">
      <button id="theme" class="theme-btn" title="Toggle light / dark" aria-label="Toggle theme">&#9680;</button>
    </div>
  </div>
  <div id="results" class="results" hidden></div>
</header>
<div class="layout">
{toc}
<main class="wrap">
  <section id="top" class="intro">
    <div class="eyebrow">PlantGen &middot; plant_gen</div>
    <h1 class="title">PlantGen documentation</h1>
    <p class="standfirst">PlantGen grows perennial plants in Blender: stems, leaves, flowers and fruit from a growth simulation, shaped by physics, with a leaf generator and a flower generator for their meshes. Start with the tutorials, look up a task in the how-to guides, read the explanations for why it works the way it does, and use the reference for every one of the {n_fields} controls.</p>
  </section>
{pages}
  <section id="reference-top">
    <div class="sec-head"><h2>Reference</h2><span class="sec-kicker">every control</span></div>
    <p class="sec-intro" id="couplings-note">Every panel of the PlantGen tab, in the order the tab shows it: each fold and sub-panel, and each control in it with its default, range and what it does. A control that only some components, levels or modes show says which. The reference is generated from the panels themselves, so it matches the add-on you have.</p>
  </section>
{reference}
</main>
</div>
<script>{js}</script>
</body>
</html>
"""
