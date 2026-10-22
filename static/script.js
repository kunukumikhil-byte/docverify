const $ = (id) => document.getElementById(id);
const state = { camera: null, pages: [] };

document.querySelectorAll(".tab").forEach(btn => btn.addEventListener("click", () => {
  document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b === btn));
  document.querySelectorAll(".panel").forEach(p => p.classList.toggle("active", p.id === `panel-${btn.dataset.panel}`));
}));
function showResult(el, html) { el.innerHTML = html; el.classList.remove("hidden"); }
function escapeHtml(s) { return String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
async function requestJson(url, options) {
  let response;
  try { response = await fetch(url, options); } catch { throw new Error("Network error. Check your connection and try again."); }
  const type = response.headers.get("content-type") || "";
  const data = type.includes("application/json") ? await response.json() : {};
  if (!response.ok || data.ok === false) throw new Error(data.error || `Request failed (${response.status}).`);
  return data;
}
function fileForm(entries) { const form = new FormData(); entries.forEach(([key, value]) => form.append(key, value)); return form; }
function buttonBusy(button, busy, label) { if (busy) { button.dataset.label = button.textContent; button.textContent = label || "Working…"; button.disabled = true; } else { button.textContent = button.dataset.label || button.textContent; button.disabled = false; } }
function renderComparison(el, data) {
  const passages = data.matching_passages || [];
  let html = `<div class="score">${Number(data.similarity_percent || 0).toFixed(2)}%</div><p><strong>Text similarity</strong></p>`;
  if (data.word_count_a != null) html += `<p class="file-label">${data.word_count_a} words vs ${data.word_count_b} words</p>`;
  html += `<p class="fineprint">${escapeHtml(data.note || "")}</p>`;
  if (passages.length) html += `<div class="passages-title">Matching passages (${passages.length})</div>` + passages.map(p => `<div class="passage"><strong>${p.words} matching words</strong><br>${escapeHtml(p.text)}</div>`).join("");
  else html += `<p>No long matching passages were detected.</p>`;
  showResult(el, html);
}
$("aiText").addEventListener("input", () => {
  $("aiCharCount").textContent = `${$("aiText").value.length.toLocaleString()} characters`;
});
$("aiFileInput").addEventListener("change", () => {
  $("aiFileLabel").textContent = $("aiFileInput").files[0]?.name || "No document selected";
});
$("detectFileBtn").addEventListener("click", async e => {
  const btn = e.currentTarget;
  buttonBusy(btn, true, "Extracting and analyzing…");
  try {
    const file = $("aiFileInput").files[0];
    if (!file) throw new Error("Choose a PDF or document first.");
    const form = new FormData();
    form.append("file", file);
    const data = await requestJson("/api/ai-detect", {method: "POST", body: form});
    const ai = Number(data.ai_likelihood_percent || 0);
    const human = Number(data.human_likelihood_percent || 0);
    const label = data.verdict || "Inconclusive";
    showResult($("aiResult"), `<div class="score">${ai.toFixed(1)}%</div><p><strong>AI-likelihood estimate</strong></p><p>Human-likelihood estimate: <strong>${human.toFixed(1)}%</strong></p><p><strong>${escapeHtml(label)}</strong></p><p class="fineprint">File: ${escapeHtml(file.name)} · Model: ${escapeHtml(data.model || "local trained detector")}</p><p>${escapeHtml(data.note || "")}</p>`);
  } catch (err) {
    showResult($("aiResult"), `<h3>Could not analyze document</h3><p>${escapeHtml(err.message)}</p><p class="fineprint">Scanned PDFs with no selectable text require OCR. Detection is an estimate, not proof of authorship.</p>`);
  } finally {
    buttonBusy(btn, false);
  }
});
$("detectAiBtn").addEventListener("click", async e => {
  const btn = e.currentTarget;
  buttonBusy(btn, true, "Analyzing…");
  try {
    const data = await requestJson("/api/ai-detect", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({text: $("aiText").value})
    });
    const ai = Number(data.ai_likelihood_percent || 0);
    const human = Number(data.human_likelihood_percent || 0);
    const label = data.verdict || (ai > human ? "More AI-like signals" : human > ai ? "More human-like signals" : "Inconclusive");
    showResult($("aiResult"), `<div class="score">${ai.toFixed(1)}%</div><p><strong>AI-likelihood estimate</strong></p><p>Human-likelihood estimate: <strong>${human.toFixed(1)}%</strong></p><p><strong>${escapeHtml(label)}</strong></p><p class="fineprint">Model: ${escapeHtml(data.model || "local trained detector")}</p><p>${escapeHtml(data.note || "")}</p>`);
  } catch(err) {
    showResult($("aiResult"), `<h3>Could not analyze text</h3><p>${escapeHtml(err.message)}</p><p class="fineprint">No result was fabricated. If the model is missing, train it using the included Google Colab notebook, add models/ai_detector.joblib to the project, and redeploy.</p>`);
  } finally {
    buttonBusy(btn, false);
  }
});
$("imagesInput").addEventListener("change",()=>{$("imagesLabel").textContent=[...$("imagesInput").files].map(f=>f.name).join(", ")||"No images selected";});
$("pdfInput").addEventListener("change",()=>{$("pdfLabel").textContent=$("pdfInput").files[0]?.name||"No PDF selected";});
$("extractInput").addEventListener("change",()=>{$("extractLabel").textContent=$("extractInput").files[0]?.name||"No file selected";});
async function downloadResponse(url, form, fallbackName, resultEl) {
  let response;
  try { response=await fetch(url,{method:"POST",body:form}); } catch { throw new Error("Network error. Check your connection and try again."); }
  const type=response.headers.get("content-type")||"";
  if(!response.ok || type.includes("application/json")) {
    const data=type.includes("application/json")?await response.json():{};
    throw new Error(data.error||`Conversion failed (${response.status}).`);
  }
  const blob=await response.blob();
  if(!blob.size) throw new Error("The server returned an empty file.");
  const disposition=response.headers.get("content-disposition")||"";
  const match=disposition.match(/filename="?([^";]+)"?/i);
  const name=match?.[1]||fallbackName;
  const urlObject=URL.createObjectURL(blob), a=document.createElement("a");
  a.href=urlObject;a.download=name;document.body.appendChild(a);a.click();a.remove();
  setTimeout(()=>URL.revokeObjectURL(urlObject),1000);
  showResult(resultEl,`<p><strong>Ready:</strong> ${escapeHtml(name)}</p><p>Your download should begin automatically.</p>`);
}
$("imagesPdfBtn").addEventListener("click",async e=>{
  const btn=e.currentTarget;buttonBusy(btn,true);
  try {
    const files=[...$("imagesInput").files]; if(!files.length) throw new Error("Choose at least one image.");
    await downloadResponse("/api/convert/images-to-pdf",fileForm(files.map(f=>["files",f])),"docverify-scan.pdf",$("convertResult"));
  }catch(err){showResult($("convertResult"),`<p>${escapeHtml(err.message)}</p>`);}finally{buttonBusy(btn,false);}
});
$("pdfImagesBtn").addEventListener("click",async e=>{
  const btn=e.currentTarget;buttonBusy(btn,true);
  try {const f=$("pdfInput").files[0];if(!f)throw new Error("Choose a PDF first.");await downloadResponse("/api/convert/pdf-to-images",fileForm([["file",f]]),"docverify-pages.zip",$("convertResult"));}
  catch(err){showResult($("convertResult"),`<p>${escapeHtml(err.message)}</p>`);}finally{buttonBusy(btn,false);}
});
$("extractBtn").addEventListener("click",async e=>{
  const btn=e.currentTarget;buttonBusy(btn,true);
  try {
    const f=$("extractInput").files[0];if(!f)throw new Error("Choose a document first.");
    const data=await requestJson("/api/extract",{method:"POST",body:fileForm([["file",f]])});
    if(!data.text.trim()) throw new Error("No text found. Image-only PDFs and photos need OCR, which is not enabled in this version.");
    showResult($("extractResult"),`<h3>${escapeHtml(data.filename)}</h3><p class="file-label">${data.characters} characters extracted</p><textarea id="extractedText" rows="12" readonly style="width:100%;padding:12px;border:1px solid #e4e9f1;border-radius:8px">${escapeHtml(data.text)}</textarea><button class="secondary" id="copyExtracted">Copy text</button>`);
    $("copyExtracted").addEventListener("click",async()=>{try{await navigator.clipboard.writeText(data.text);$("copyExtracted").textContent="Copied";}catch{$("copyExtracted").textContent="Copy unavailable";}});
  } catch(err){showResult($("extractResult"),`<p>${escapeHtml(err.message)}</p>`);}finally{buttonBusy(btn,false);}
});

$("plagCompareBtn").addEventListener("click",async e=>{const btn=e.currentTarget;buttonBusy(btn,true,"Comparing…");try{const a=$("plagFile1").files[0],b=$("plagFile2").files[0];if(!a||!b)throw new Error("Choose both documents first.");const data=await requestJson("/api/plagiarism/compare",{method:"POST",body:fileForm([["file1",a],["file2",b]])});renderComparison($("plagResult"),data);}catch(err){showResult($("plagResult"),`<h3>Could not compare documents</h3><p>${escapeHtml(err.message)}</p>`);}finally{buttonBusy(btn,false);}});
$("convertDocInput").addEventListener("change",()=>{$("convertDocLabel").textContent=$("convertDocInput").files[0]?.name||"No document selected";});
$("convertDocBtn").addEventListener("click",async e=>{const btn=e.currentTarget;buttonBusy(btn,true,"Converting…");try{const f=$("convertDocInput").files[0];if(!f)throw new Error("Choose a document first.");await downloadResponse("/api/convert/document",fileForm([["file",f],["target",$("convertTarget").value]]),"converted-document",$("convertResult"));}catch(err){showResult($("convertResult"),`<p>${escapeHtml(err.message)}</p>`);}finally{buttonBusy(btn,false);}});
$("cropInput").addEventListener("change",()=>{const f=$("cropInput").files[0];$("cropLabel").textContent=f?.name||"No file selected";$("cropPageWrap").classList.toggle("hidden",!!f&&!f.name.toLowerCase().endsWith(".pdf"));});
$("cropBtn").addEventListener("click",async e=>{const btn=e.currentTarget;buttonBusy(btn,true,"Cropping…");try{const f=$("cropInput").files[0];if(!f)throw new Error("Choose a PDF or image first.");const entries=[["file",f],["left",$("cropLeft").value],["top",$("cropTop").value],["right",$("cropRight").value],["bottom",$("cropBottom").value]];const isPdf=f.name.toLowerCase().endsWith(".pdf");if(isPdf)entries.push(["page",$("cropPage").value.trim()||"all"]);await downloadResponse(isPdf?"/api/crop/pdf":"/api/crop/image",fileForm(entries),"cropped-file",$("cropResult"));}catch(err){showResult($("cropResult"),`<p>${escapeHtml(err.message)}</p>`);}finally{buttonBusy(btn,false);}});

function updatePages(){
  $("pageCount").textContent=`${state.pages.length} page${state.pages.length===1?"":"s"}`;
  $("scanPdfBtn").disabled=!state.pages.length;$("clearPagesBtn").disabled=!state.pages.length;
  $("pageThumbs").replaceChildren();
  state.pages.forEach((page,index)=>{const box=document.createElement("div");box.className="thumb";const img=document.createElement("img");img.src=page.url;img.alt=`Scanned page ${index+1}`;const remove=document.createElement("button");remove.textContent="×";remove.setAttribute("aria-label",`Remove page ${index+1}`);remove.addEventListener("click",()=>{URL.revokeObjectURL(state.pages[index].url);state.pages.splice(index,1);updatePages();});box.append(img,remove);$("pageThumbs").append(box);});
}
$("startCameraBtn").addEventListener("click",async()=>{
  if(!navigator.mediaDevices?.getUserMedia){$("cameraStatus").textContent="Camera is not supported by this browser. You can still use the image upload converter.";return;}
  try {
    state.camera=await navigator.mediaDevices.getUserMedia({video:{facingMode:{ideal:"environment"}},audio:false});
    $("cameraVideo").srcObject=state.camera;$("cameraVideo").classList.add("running");$("cameraPlaceholder").classList.add("hidden");
    $("captureBtn").disabled=false;$("stopCameraBtn").disabled=false;$("cameraStatus").textContent="Camera is running.";
  }catch(err){$("cameraStatus").textContent=err.name==="NotAllowedError"?"Camera permission was denied. Allow access in browser settings.":"Could not start the camera. Check that no other app is using it.";}
});
$("stopCameraBtn").addEventListener("click",()=>{
  state.camera?.getTracks().forEach(t=>t.stop());state.camera=null;$("cameraVideo").srcObject=null;$("cameraVideo").classList.remove("running");$("cameraPlaceholder").classList.remove("hidden");$("captureBtn").disabled=true;$("stopCameraBtn").disabled=true;$("cameraStatus").textContent="Camera stopped.";
});
$("captureBtn").addEventListener("click",()=>{
  const video=$("cameraVideo");if(!video.videoWidth||!video.videoHeight){$("cameraStatus").textContent="Wait for the camera preview, then try again.";return;}
  const canvas=$("captureCanvas");canvas.width=video.videoWidth;canvas.height=video.videoHeight;canvas.getContext("2d").drawImage(video,0,0);
  canvas.toBlob(blob=>{if(!blob){$("cameraStatus").textContent="Capture failed. Try again.";return;}state.pages.push({blob,url:URL.createObjectURL(blob)});updatePages();$("cameraStatus").textContent=`Captured page ${state.pages.length}.`;}, "image/jpeg", .92);
});
$("scanPdfBtn").addEventListener("click",async e=>{
  const btn=e.currentTarget;buttonBusy(btn,true);
  try {
    const form=new FormData();state.pages.forEach((p,i)=>form.append("files",p.blob,`scan-${i+1}.jpg`));
    await downloadResponse("/api/convert/images-to-pdf",form,"docverify-scan.pdf",$("cameraResult"));
  }catch(err){showResult($("cameraResult"),`<p>${escapeHtml(err.message)}</p>`);}finally{buttonBusy(btn,false);}
});
$("clearPagesBtn").addEventListener("click",()=>{state.pages.forEach(p=>URL.revokeObjectURL(p.url));state.pages=[];updatePages();$("cameraResult").classList.add("hidden");});
window.addEventListener("beforeunload",()=>{state.pages.forEach(p=>URL.revokeObjectURL(p.url));state.camera?.getTracks().forEach(t=>t.stop());});
