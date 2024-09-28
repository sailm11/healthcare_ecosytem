const modelUrl = '/staticFolder/models/scene.gltf';
const modelSection = document.getElementById('model-section');

// Create a scene, camera, and renderer
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, modelSection.clientWidth / modelSection.clientHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(modelSection.clientWidth, modelSection.clientHeight); // Fixed typo: clientWidt to clientWidth
modelSection.appendChild(renderer.domElement);
modelSection.addEventListener('click', onClick, false);



let currentRotation = 0;
let hotspots = [];

scene.background = new THREE.Color(0xf6f5ff)

// Add lighting
const light = new THREE.DirectionalLight(0xf6f5ff, 1);
light.position.set(5, 5, 5).normalize();
scene.add(light);

const ambientLight = new THREE.AmbientLight(0x404040);  
scene.add(ambientLight);

// Load the 3D Human Body model using GLTFLoader
const loader = new THREE.GLTFLoader();
loader.load(modelUrl, function (gltf) {
    const model = gltf.scene;
    model.position.set(0, -1.5, 0); // Adjust position
    model.scale.set(0.015, 0.015, 0.015); // Adjust size
    scene.add(model);

    addHotspots();

    model.traverse((child) => {
        if (child.isMesh) {
            if (child.name === '') {
                child.name = 'UnnamedPart';
            }
            // For testing: you could manually assign names to parts
            if (child.name.includes('head')) child.name = 'Head';
            if (child.name.includes('leg')) child.name = 'Leg';
            // Continue assigning names to body parts based on the structure
        }
    });

    // Render the scene with animation
    function animate() {
        requestAnimationFrame(animate);
        if (model) {
            model.rotation.y = currentRotation;  // Apply rotation from slider
        }
        renderer.render(scene, camera);
    }

    animate();
}, undefined, function (error) {
    console.error('An error occurred while loading the model', error);
});

// Slider for rotation
const slider = document.getElementById('rotation-slider');
slider.addEventListener('input', function () {
    const rotationValue = (slider.value - 180) * (Math.PI / 180); // Convert degrees to radians
    rotateModel(rotationValue);
});

// Rotate the model using the slider value
function rotateModel(rotationValue) {
    currentRotation = rotationValue;
}

// Set camera position
camera.position.z = 3;

// Ensure the model is visible on load
function adjustCameraAndRender() {
    camera.position.z = 3; // Adjust this value if needed
    renderer.render(scene, camera);
}

function addHotspots() {
    // Example hotspot positions on the model (change the coordinates accordingly)
    const hotspotCoordinates = [
        { x: 0.5, y: 1.5, z: 0 }, // Hotspot 1        
        { x: -1, y: 1.2, z: 0.8 }, // Hotspot 2
        { x: 0, y: 0.5, z: -1 }, // Hotspot 3
    ];

    const hotspotMaterial = new THREE.MeshBasicMaterial({ color: 0xff0000 }); // Red color for visibility
    const hotspotGeometry = new THREE.SphereGeometry(0.05, 32, 32); // Small sphere for hotspots

    hotspotCoordinates.forEach((pos) => {
        const hotspot = new THREE.Mesh(hotspotGeometry, hotspotMaterial);
        // console.log(hotspot)
        hotspot.position.set(pos.x, pos.y, pos.z); // Set position
        hotspots.push(hotspot);
        scene.add(hotspot);
    });
}

const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();

function onClick(event) {
    // Calculate mouse position in normalized device coordinates (-1 to +1)
    const rect = modelSection.getBoundingClientRect();

    mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

    // Update the raycaster with the camera and mouse position
    raycaster.setFromCamera(mouse, camera);

    // Check if any hotspot is clicked
    const intersects = raycaster.intersectObjects(hotspots);
    console.log(intersects)

    if (intersects.length > 0) {
        const clickedHotspot = intersects[0].object;
        console.log("Hotspot clicked!", clickedHotspot.position);
        // You can show information, trigger events, etc., here
        alert("You clicked on a hotspot at position: " + JSON.stringify(clickedHotspot.position));
    }
}

// Handle window resize
window.addEventListener('resize', function () {
    const width = modelSection.clientWidth;
    const height = modelSection.clientHeight;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    adjustCameraAndRender(); // Call to render after resizing
});

// Initial render to ensure visibility
adjustCameraAndRender();