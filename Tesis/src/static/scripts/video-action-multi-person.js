var deteccionesTiempoReal = [];
var framesSkipToAnalyze = 3; // por defecto
var selDimension = '2D';
var detecciones = [];
var filename = '';
var source = null;

document.addEventListener('DOMContentLoaded', function () {
    const uploadForm = document.getElementById('upload-form');
    const videoFile = document.getElementById('video-file');
    const uploadBtn = document.getElementById('upload-btn');
    const reuploadBtn = document.getElementById('reupload-btn');
    const initInfo = document.getElementById('init-info');
    const progressContainer = document.getElementById('progress-container');
    const processedVideo = document.getElementById('full-video');
    const timestampsList = document.getElementById('timestamps-list');
    const imgProcessed = document.getElementById('img-processed-video');
    const videoPreview = document.getElementById("loaded-video");
    const skipframes_ddMenuButton = document.getElementById("dropdown1");
    const skipframes_ddItems = document.querySelectorAll('#dd1 .dropdown-item');
    const dimension_ddMenuButton = document.getElementById("dropdown2");
    const dimension_ddItems = document.querySelectorAll('#dd2 .dropdown-item');
    const labelWarning = document.getElementById('label-warning');
    const loadVideoInfo = document.getElementById('load-video-info');
    const viewResultsBtn = document.getElementById('view-results');
    const closeModalBtn = document.getElementById('close-btn');
    const fullVideo = document.getElementById('full-video');
    const progressBar = document.getElementsByClassName('progress')[0];
    const modal = document.getElementById('myModal');
    const realtime_count = document.getElementById('realtime-results-count');
    const realtime_list = document.getElementById('realtime-list');
    const policeImage = document.getElementById('police-image');
    const irSubContainer = document.getElementById('ir-info-subcontainer');
    const foundResults = document.getElementById('found-results');
    const realtimeSpinner = document.getElementById('realtime-spinner');

    uploadForm.addEventListener('submit', function (e) {
        e.preventDefault();
        if (!videoFile.files[0]) {
            showToast("Primero seleccione un archivo de video para continuar", "info");
            return;
        }

        const formData = new FormData();
        formData.append('video', videoFile.files[0]);

        fetch('/save-video-lstm', { method: 'POST', body: formData })
            .then(async res => {
                const data = await res.json();
                if (!res.ok && res.status !== 409) throw data;
                filename = data.filename || videoFile.files[0].name;

                // Bloquear controles mientras procesa
                skipframes_ddMenuButton.disabled = true;
                dimension_ddMenuButton.disabled = true;
                uploadBtn.disabled = true;
                videoFile.disabled = true;
                irSubContainer.style.display = 'none';
                foundResults.style.display = 'block';

                // Iniciar SSE
                startSSE(filename);
                showToast("Procesando video", "info");
            })
            .catch(err => {
                showToast("Error al procesar el video", "error");
            });
    });

function startSSE(filename) {
    if (source) source.close();
    let init = false;
    realtimeSpinner.style.display = 'block';

    // Abrir SSE hacia Flask
    source = new EventSource(`/evaluate-final/${filename}/${framesSkipToAnalyze}`);

    source.onmessage = function (event) {
        if (event.data === "EOF") {
            source.close();
            realtimeSpinner.style.display = 'none';
            showToast("Video procesado correctamente", "success");

            // 🔑 Reactivar controles para reprocesar un nuevo video
            reuploadBtn.style.display = 'inline-block';
            uploadBtn.disabled = false;
            videoFile.disabled = false;

            // Cargar directamente los archivos generados por evaluate_final
            const videoUrl = `/static/videos/results/processed_${filename.replace('.mp4','.webm')}`;
            const jsonUrl = `/static/videos/results/results_${filename.replace('.mp4','.json')}`;

            setProcessedVideo(videoUrl);

            fetch(jsonUrl)
                .then(res => res.json())
                .then(data => {
                    detecciones = data;
                    displayDetections();
                })
                .catch(err => {
                    showToast("No se encontraron resultados", "error");
                });

            return;
        }

        const data = JSON.parse(event.data);

        if (!init) {
            progressContainer.style.display = 'block';
            imgProcessed.style.display = 'block';
            initInfo.style.display = 'none';
            videoPreview.play();
            init = true;
        }

        updateProgressBar(data.progress);
        if (data.frame) {
            imgProcessed.src = 'data:image/jpeg;base64,' + data.frame;
        }
        if (data.detections && data.detections.length) {
            updateRealTimeDetections(data.detections);
        }
    };

    source.onerror = function () {
        showToast("Error en el stream de progreso", "error");
        source.close();
    };
}



    videoFile.addEventListener("change", function (event) {
        const file = event.target.files[0];
        if (file) {
            loadVideoInfo.style.display = 'none';
            videoPreview.style.display = 'block';
            videoPreview.src = URL.createObjectURL(file);
            videoPreview.load();
            filename = file.name;
        }
    });

reuploadBtn.addEventListener('click', function () {
    // Reset UI
    detecciones = [];
    deteccionesTiempoReal = [];
    timestampsList.innerHTML = '';
    realtime_list.innerHTML = '';
    realtime_count.innerText = '0';
    progressBar.innerHTML = '';
    imgProcessed.src = '';
    initInfo.style.display = 'block';
    progressContainer.style.display = 'none';
    foundResults.style.display = 'none';
    irSubContainer.style.display = 'block';
    skipframes_ddMenuButton.disabled = false;
    dimension_ddMenuButton.disabled = false;
    uploadBtn.disabled = false;
    videoFile.disabled = false;

    // 🔑 Limpieza de variables
    filename = '';
    videoFile.value = '';

    if (source) source.close();
    showToast("Listo para procesar un nuevo video", "success");
});


    skipframes_ddItems.forEach(item => {
        item.addEventListener('click', function (e) {
            e.preventDefault();
            let selectedText = this.textContent.trim();
            if (!selectedText.includes('Sin saltos')) {
                framesSkipToAnalyze = parseInt(selectedText.match(/\d+/)[0]);
                if (selectedText.includes("defecto")) selectedText = selectedText.replace(' (Por defecto)', '');
                skipframes_ddMenuButton.innerHTML = `<i class="fa-solid fa-sliders"></i> Analizar cada: ${selectedText}&nbsp;`;
            } else {
                framesSkipToAnalyze = 0;
                skipframes_ddMenuButton.innerHTML = `<i class="fa-solid fa-sliders"></i> ${selectedText}&nbsp;`;
            }
            labelWarning.style.display = framesSkipToAnalyze < 3 ? 'block' : 'none';
        });
    });

    dimension_ddItems.forEach(item => {
        item.addEventListener('click', function (e) {
            e.preventDefault();
            let selectedText = this.textContent.trim();
            selDimension = selectedText;
            dimension_ddMenuButton.innerHTML = `<i class="fa-solid fa-male"></i> Estimación postural: ${selectedText}&nbsp;`;
        });
    });

    viewResultsBtn.addEventListener('click', function () {
        if (!detecciones.length) {
            showToast("No hay resultados para mostrar", "info");
            return;
        }
        modal.style.display = "block";
    });

    closeModalBtn.addEventListener('click', function () {
        modal.style.display = "none";
    });

    document.addEventListener('keyup', function (event) {
        if (event.key === 'Escape' && modal.style.display === 'block') {
            modal.style.display = "none";
        }
    });

    function setProcessedVideo(videoUrl) {
        if (!videoUrl || videoUrl === 'undefined') {
            showToast("Ruta de video inválida", "error");
            return;
        }
        fullVideo.src = videoUrl;
        fullVideo.load();
    }

    function updateProgressBar(percentage) {
        if (percentage >= 100) document.getElementById('progress-text').style.display = 'none';
        progressBar.innerHTML = `<div class="progress-bar progress-bar-striped ${percentage >= 100 ? 'bg-success' : ''}" role="progressbar"
            style="width: ${percentage}%;" aria-valuenow="${percentage}" aria-valuemin="0" aria-valuemax="100">${percentage}%</div>`;
    }

    function displayDetections() {
    timestampsList.innerHTML = '';
    if (detecciones.length === 0) {
        timestampsList.innerHTML = '<p class="text-center">No se detectaron eventos.</p>';
        return;
    }

    detecciones.forEach((evento, idx) => {
        const item = document.createElement('div');
        item.className = 'timestamp-item';
        item.dataset.time = evento.inicio_segundo;

        const formattedTime = `${evento.inicio_segundo.toFixed(2)}s - ${evento.fin_segundo.toFixed(2)}s`;

        let tagHTML = '';
        if (evento.tipo_evento === 'PELEAR') {
            tagHTML = `<span class="behavior-tag peleando">Pelea</span>`;
        } else if (evento.tipo_evento === 'DISTURBIO') {
            tagHTML = `<span class="behavior-tag disturbio">Disturbio</span>`;
        }

        item.innerHTML = `<div><strong>Evento ${idx + 1}</strong> (${formattedTime}) - ${tagHTML} | Precisión máx: ${evento.precision_maxima.toFixed(2)}</div>`;
        item.addEventListener('click', function () {
            processedVideo.currentTime = evento.inicio_segundo;
            processedVideo.play();
        });

        timestampsList.appendChild(item);
    });
}



    function updateRealTimeDetections(detections) {
        deteccionesTiempoReal = [...new Set([...deteccionesTiempoReal, ...detections])];
        if (!policeImage.src.includes('angry')) policeImage.src = '/static/assets/angry_police.jpg';

        if (deteccionesTiempoReal.length > parseInt(realtime_count.innerText)) {
            realtime_count.innerText = deteccionesTiempoReal.length;
            const ultimaDeteccion = detections[detections.length - 1];
            if (ultimaDeteccion === 'PELEAR') {
                realtime_list.innerHTML += `${realtime_list.innerText !== '' ? '<br>' : ''}<label><i class="fa-solid fa-user"></i>Pelea detectada</label>`;
            } else if (ultimaDeteccion === 'DISTURBIO') {
                realtime_list.innerHTML += `${realtime_list.innerText !== '' ? '<br>' : ''}<label><i class="fa-solid fa-users"></i>Disturbio detectado</label>`;
            }
        }
    }
});
