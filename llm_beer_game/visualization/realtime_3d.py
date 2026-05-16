import streamlit as st
import streamlit.components.v1 as components
import json
import sys
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import time

# Add analysis module path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'analysis'))

class RealtimeSupplyChain3D:
    """Real-time 3D Supply Chain Visualization Component"""
    
    def __init__(self):
        self.is_active = False
        self.current_round = 0
        self.simulation_data = []
        self.container = None
        self.previous_data = None  # For animation detection
        

        
    def initialize_display(self, container):
        """Initialize 3D display container"""
        self.container = container
        self.is_active = True
        self._initialized = False
        
        try:
            # Create initial 3D scene HTML
            initial_html = self._create_initial_scene()
            with self.container:
                components.html(initial_html, height=1000, scrolling=False)
            # Mark as initialized
            self._initialized = True
        except Exception as e:
            st.error(f"3D visualization initialization failed: {str(e)}")
            self.is_active = False
            self._initialized = False
            with self.container:
                st.error("3D visualization failed to load, please check network connection or refresh the page")
        
    def update_round_data(self, round_data: Dict[str, Any]):
        """Update current round data"""
        if not self.is_active or not self.container:
            return

        try:
            self.current_round = round_data.get('round', 0)

            # Ensure simulation_data is a list
            if not isinstance(self.simulation_data, list):
                st.error(f"Data structure error: simulation_data should be a list, but got {type(self.simulation_data)}")
                self.simulation_data = []

            self.simulation_data.append(round_data)

            # Remove overreaction analysis related history data updates

            # Use JavaScript to update data, avoid re-rendering the entire component
            update_script = self._create_update_script(round_data)

            # Only perform JavaScript updates after initialization
            if hasattr(self, '_initialized') and self._initialized:
                # Update existing scene via JavaScript
                with self.container:
                    components.html(f"""
                    <script>
                    {update_script}
                    </script>
                    """, height=0)
            else:
                # If not yet initialized, re-render full scene
                updated_html = self._create_updated_scene(round_data)
                with self.container:
                    components.html(updated_html, height=1000, scrolling=False)

        except Exception as e:
            st.error(f"3D visualization update failed: {str(e)}")
            # Try to reinitialize
            try:
                self._initialized = False
                initial_html = self._create_initial_scene()
                with self.container:
                    components.html(initial_html, height=1000, scrolling=False)
                self._initialized = True
            except:
                st.error("3D visualization reinitialization failed, please refresh the page")
        
    def toggle_display(self, enable: bool):
        """Toggle display state"""
        self.is_active = enable
        if not enable and self.container:
            self.container.empty()
            
    def reset(self):
        """Reset visualization state"""
        self.current_round = 0
        self.simulation_data = []
        self._initialized = False
        # Remove agent_history reset
        if hasattr(self, 'previous_data'):
            delattr(self, 'previous_data')
        if self.container:
            self.container.empty()
    
    # Remove overreaction indicator calculation method
            
    def get_html(self) -> str:
        """Get HTML of current 3D scene"""
        if self.simulation_data:
            # If data exists, return updated scene
            return self._create_updated_scene(self.simulation_data[-1])
        else:
            # If no data, return initial scene
            return self._create_initial_scene()
            
    def update_data(self, round_data: Dict[str, Any]):
        """Update simulation data"""
        self.current_round = round_data.get('round', 0)
        self.simulation_data.append(round_data)
        # Limit data history length to avoid excessive memory usage
        if len(self.simulation_data) > 50:
            self.simulation_data = self.simulation_data[-50:]
            
    def _create_initial_scene(self) -> str:
        """Create initial 3D scene HTML"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Real-Time Supply Chain 3D Visualization</title>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
            <style>
                body {{
                    margin: 0;
                    padding: 0;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    font-family: 'Arial', sans-serif;
                    overflow: hidden;
                }}
                #container {{
                    width: 100%;
                    height: 1000px;
                    position: relative;
                }}
                #container.fullscreen {{
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100vw;
                    height: 100vh;
                    z-index: 9999;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                }}
                #fullscreen-btn {{
                    position: absolute;
                    top: 10px;
                    left: 50%;
                    transform: translateX(-50%);
                    background: rgba(0,0,0,0.7);
                    color: white;
                    border: none;
                    padding: 10px 15px;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 14px;
                    z-index: 101;
                    transition: background 0.3s;
                }}
                #fullscreen-btn:hover {{
                    background: rgba(0,0,0,0.9);
                }}
                #exit-fullscreen-btn {{
                    position: absolute;
                    top: 20px;
                    right: 20px;
                    background: rgba(255,0,0,0.7);
                    color: white;
                    border: none;
                    padding: 10px 15px;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 14px;
                    z-index: 101;
                    display: none;
                    transition: background 0.3s;
                }}
                #exit-fullscreen-btn:hover {{
                    background: rgba(255,0,0,0.9);
                }}
                #info {{
                    position: absolute;
                    top: 10px;
                    right: 10px;
                    background: rgba(0,0,0,0.7);
                    color: white;
                    padding: 15px;
                    border-radius: 10px;
                    font-size: 14px;
                    z-index: 100;
                    min-width: 200px;
                }}
                #status {{
                    position: absolute;
                    top: 10px;
                    left: 10px;
                    background: rgba(0,128,0,0.8);
                    color: white;
                    padding: 10px;
                    border-radius: 8px;
                    font-size: 16px;
                    font-weight: bold;
                    z-index: 100;
                }}
                .metric {{
                    margin: 5px 0;
                    display: flex;
                    justify-content: space-between;
                }}
                .metric-label {{
                    font-weight: bold;
                }}
                .metric-value {{
                    color: #4CAF50;
                }}

        </head>
        <body>
            <div id="container">
                <button id="fullscreen-btn" onclick="toggleFullscreen()">🔍 Fullscreen</button>
                <button id="exit-fullscreen-btn" onclick="exitFullscreen()">❌ Exit Fullscreen</button>
<div id="status">🔄 Waiting for simulation to start...</div>
                <div id="info">
                    <div class="metric">
                        <span class="metric-label">Current Round:</span>
                        <span class="metric-value" id="current-round">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Total Inventory:</span>
                        <span class="metric-value" id="total-inventory">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Total In-Transit:</span>
                        <span class="metric-value" id="total-intransit">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Total Backorder:</span>
                        <span class="metric-value" id="total-backorder">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Market Demand:</span>
                        <span class="metric-value" id="market-demand">0</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Total Cost:</span>
                        <span class="metric-value" id="total-cost">0.00</span>
                    </div>
                </div>
            </div>
            </div>
            
            <script>
                // Initialize Three.js scene
                let scene, camera, renderer, controls;
                let agents = {{}};
                let orderArrows = [];
                
                function init() {{
                    // Create scene
                    scene = new THREE.Scene();
                    scene.background = new THREE.Color(0x87CEEB);

                    // Create camera
                    camera = new THREE.PerspectiveCamera(75, window.innerWidth / 600, 0.1, 1000);
                    camera.position.set(0, 14, 22);

                    // Create renderer
                    renderer = new THREE.WebGLRenderer({{ antialias: true }});
                    renderer.setSize(window.innerWidth, 600);
                    renderer.shadowMap.enabled = true;
                    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
                    document.getElementById('container').appendChild(renderer.domElement);

                    // Add controls
                    controls = new THREE.OrbitControls(camera, renderer.domElement);
                    controls.enableDamping = true;
                    controls.dampingFactor = 0.05;

                    // Add lighting
                    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
                    scene.add(ambientLight);

                    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
                    directionalLight.position.set(10, 10, 5);
                    directionalLight.castShadow = true;
                    scene.add(directionalLight);

                    // Add ground
                    const groundGeometry = new THREE.PlaneGeometry(80, 30);
                    const groundMaterial = new THREE.MeshLambertMaterial({{ color: 0x90EE90, transparent: true, opacity: 0.3 }});
                    const ground = new THREE.Mesh(groundGeometry, groundMaterial);
                    ground.rotation.x = -Math.PI / 2;
                    ground.receiveShadow = true;
                    scene.add(ground);

                    // Add raised platform for all participants
                    const platformGeo = new THREE.BoxGeometry(58, 0.18, 11);
                    const platformMat = new THREE.MeshLambertMaterial({{ color: 0x444444 }});
                    const platform = new THREE.Mesh(platformGeo, platformMat);
                    platform.position.set(0, -0.09, 0);
                    platform.receiveShadow = true;
                    platform.castShadow = true;
                    scene.add(platform);

                    // Platform border
                    const edgeGeo = new THREE.EdgesGeometry(platformGeo);
                    const edgeMat = new THREE.LineBasicMaterial({{ color: 0x666666 }});
                    const edgeLine = new THREE.LineSegments(edgeGeo, edgeMat);
                    edgeLine.position.copy(platform.position);
                    scene.add(edgeLine);

                    // Create supply chain nodes
                    createSupplyChainNodes();

                    // Start render loop
                    animate();
                }}
                
                function createSupplyChainNodes() {{
                    const roles = [
                        {{ name: 'customer', label: 'Customer', color: 0xFFD700, position: [-20, 0, 0] }},
                        {{ name: 'retailer', label: 'Retailer', color: 0xFF6B6B, position: [-10, 0, 0] }},
                        {{ name: 'wholesaler', label: 'Wholesaler', color: 0x4ECDC4, position: [0, 0, 0] }},
                        {{ name: 'distributor', label: 'Distributor', color: 0x45B7D1, position: [10, 0, 0] }},
                        {{ name: 'manufacturer', label: 'Manufacturer', color: 0x96CEB4, position: [20, 0, 0] }}
                    ];
                    
                    roles.forEach(role => {{
                        const group = new THREE.Group();

                        if (role.name === 'customer') {{
                            // Customer: 3D person figure
                            const headGeo = new THREE.SphereGeometry(0.3, 16, 16);
                            const headMat = new THREE.MeshLambertMaterial({{ color: 0xFFDBA4 }});
                            const head = new THREE.Mesh(headGeo, headMat);
                            head.position.y = 2.3;
                            head.castShadow = true;
                            group.add(head);

                            const bodyGeo = new THREE.CylinderGeometry(0.24, 0.32, 0.95, 8);
                            const bodyMat = new THREE.MeshLambertMaterial({{ color: role.color }});
                            const body = new THREE.Mesh(bodyGeo, bodyMat);
                            body.position.y = 1.35;
                            body.castShadow = true;
                            body.receiveShadow = true;
                            group.add(body);
                            group.userData.bodyMesh = body;

                            const leftArmGeo = new THREE.CylinderGeometry(0.08, 0.08, 0.7, 8);
                            const leftArm = new THREE.Mesh(leftArmGeo, bodyMat);
                            leftArm.position.set(-0.4, 1.5, 0);
                            leftArm.rotation.z = 0.2;
                            leftArm.castShadow = true;
                            group.add(leftArm);

                            const rightArmGeo = new THREE.CylinderGeometry(0.08, 0.08, 0.7, 8);
                            const rightArm = new THREE.Mesh(rightArmGeo, bodyMat);
                            rightArm.position.set(0.4, 1.5, 0);
                            rightArm.rotation.z = -0.2;
                            rightArm.castShadow = true;
                            group.add(rightArm);

                            const legsMat = new THREE.MeshLambertMaterial({{ color: 0x335577 }});
                            const leftLegGeo = new THREE.CylinderGeometry(0.12, 0.12, 0.8, 8);
                            const leftLeg = new THREE.Mesh(leftLegGeo, legsMat);
                            leftLeg.position.set(-0.14, 0.45, 0);
                            leftLeg.castShadow = true;
                            group.add(leftLeg);

                            const rightLegGeo = new THREE.CylinderGeometry(0.12, 0.12, 0.8, 8);
                            const rightLeg = new THREE.Mesh(rightLegGeo, legsMat);
                            rightLeg.position.set(0.14, 0.45, 0);
                            rightLeg.castShadow = true;
                            group.add(rightLeg);
                        }} else if (role.name === 'manufacturer') {{
                            // Factory: building with chimney and smoke
                            const bodyGeo = new THREE.BoxGeometry(1.5, 1.15, 1.5);
                            const bodyMat = new THREE.MeshLambertMaterial({{ color: role.color }});
                            const body = new THREE.Mesh(bodyGeo, bodyMat);
                            body.position.y = 0.575;
                            body.castShadow = true;
                            body.receiveShadow = true;
                            group.add(body);
                            group.userData.bodyMesh = body;
                            const roofGeo = new THREE.ConeGeometry(0.85, 0.45, 4);
                            const roofMat = new THREE.MeshLambertMaterial({{ color: 0x555555 }});
                            const roof = new THREE.Mesh(roofGeo, roofMat);
                            roof.position.y = 1.35;
                            roof.rotation.y = Math.PI / 4;
                            roof.castShadow = true;
                            group.add(roof);
                            const chimneyGeo = new THREE.CylinderGeometry(0.13, 0.14, 0.8, 8);
                            const chimneyMat = new THREE.MeshLambertMaterial({{ color: 0x888888 }});
                            const chimney = new THREE.Mesh(chimneyGeo, chimneyMat);
                            chimney.position.set(0.4, 1.5, 0.27);
                            chimney.castShadow = true;
                            group.add(chimney);
                            for (let si = 0; si < 3; si++) {{
                                const sGeo = new THREE.SphereGeometry(0.08 + si * 0.04, 8, 8);
                                const sMat = new THREE.MeshLambertMaterial({{ color: 0xcccccc, transparent: true, opacity: 0.6 - si * 0.15 }});
                                const sMesh = new THREE.Mesh(sGeo, sMat);
                                sMesh.position.set(0.35 + si * 0.04, 1.78 + si * 0.18, 0.24);
                                group.add(sMesh);
                            }}
                        }} else if (role.name === 'retailer') {{
                            // Store/shop: building with awning, door, and sign
                            const bodyGeo = new THREE.BoxGeometry(1.5, 1.15, 1.25);
                            const bodyMat = new THREE.MeshLambertMaterial({{ color: role.color }});
                            const body = new THREE.Mesh(bodyGeo, bodyMat);
                            body.position.y = 0.575;
                            body.castShadow = true;
                            body.receiveShadow = true;
                            group.add(body);
                            group.userData.bodyMesh = body;
                            const roofGeo = new THREE.ConeGeometry(0.9, 0.45, 4);
                            const roofMat = new THREE.MeshLambertMaterial({{ color: 0x8B4513 }});
                            const roof = new THREE.Mesh(roofGeo, roofMat);
                            roof.position.y = 1.35;
                            roof.rotation.y = Math.PI / 4;
                            roof.castShadow = true;
                            group.add(roof);
                            const awningGeo = new THREE.BoxGeometry(1.0, 0.07, 0.35);
                            const awningMat = new THREE.MeshLambertMaterial({{ color: 0x444444 }});
                            const awning = new THREE.Mesh(awningGeo, awningMat);
                            awning.position.set(0, 0.55, 0.8);
                            group.add(awning);
                            const doorGeo = new THREE.BoxGeometry(0.35, 0.48, 0.06);
                            const doorMat = new THREE.MeshLambertMaterial({{ color: 0x8B4513 }});
                            const door = new THREE.Mesh(doorGeo, doorMat);
                            door.position.set(0, 0.24, 0.58);
                            group.add(door);
                            const signGeo = new THREE.BoxGeometry(0.42, 0.2, 0.06);
                            const signMat = new THREE.MeshLambertMaterial({{ color: 0xFFD700 }});
                            const sign = new THREE.Mesh(signGeo, signMat);
                            sign.position.set(0, 0.76, 0.58);
                            group.add(sign);
                        }} else if (role.name === 'wholesaler') {{
                            // Warehouse: wide building with loading docks
                            const bodyGeo = new THREE.BoxGeometry(1.9, 1.05, 1.6);
                            const bodyMat = new THREE.MeshLambertMaterial({{ color: role.color }});
                            const body = new THREE.Mesh(bodyGeo, bodyMat);
                            body.position.y = 0.525;
                            body.castShadow = true;
                            body.receiveShadow = true;
                            group.add(body);
                            group.userData.bodyMesh = body;
                            const roofGeo = new THREE.BoxGeometry(2.0, 0.09, 1.75);
                            const roofMat = new THREE.MeshLambertMaterial({{ color: 0x666666 }});
                            const roof = new THREE.Mesh(roofGeo, roofMat);
                            roof.position.y = 1.06;
                            roof.castShadow = true;
                            group.add(roof);
                            for (let di = -1; di <= 1; di++) {{
                                const dockGeo = new THREE.BoxGeometry(0.35, 0.25, 0.2);
                                const dockMat = new THREE.MeshLambertMaterial({{ color: 0x333333 }});
                                const dock = new THREE.Mesh(dockGeo, dockMat);
                                dock.position.set(di * 0.42, 0.17, 0.76);
                                group.add(dock);
                                const doorGeo = new THREE.BoxGeometry(0.29, 0.28, 0.06);
                                const doorMat = new THREE.MeshLambertMaterial({{ color: 0x777777 }});
                                const door = new THREE.Mesh(doorGeo, doorMat);
                                door.position.set(di * 0.42, 0.45, 0.75);
                                group.add(door);
                            }}
                            for (let vi = -1; vi <= 1; vi += 2) {{
                                const ventGeo = new THREE.CylinderGeometry(0.08, 0.08, 0.28, 8);
                                const ventMat = new THREE.MeshLambertMaterial({{ color: 0x999999 }});
                                const vent = new THREE.Mesh(ventGeo, ventMat);
                                vent.position.set(vi * 0.5, 1.14, 0.24);
                                group.add(vent);
                            }}
                        }} else if (role.name === 'distributor') {{
                            // Distribution center: building with office tower, loading bays, and antenna
                            const bodyGeo = new THREE.BoxGeometry(1.7, 1.05, 1.7);
                            const bodyMat = new THREE.MeshLambertMaterial({{ color: role.color }});
                            const body = new THREE.Mesh(bodyGeo, bodyMat);
                            body.position.y = 0.525;
                            body.castShadow = true;
                            body.receiveShadow = true;
                            group.add(body);
                            group.userData.bodyMesh = body;
                            const roofGeo = new THREE.BoxGeometry(1.82, 0.09, 1.82);
                            const roofMat = new THREE.MeshLambertMaterial({{ color: 0x666666 }});
                            const roof = new THREE.Mesh(roofGeo, roofMat);
                            roof.position.y = 1.06;
                            roof.castShadow = true;
                            group.add(roof);
                            const towerGeo = new THREE.BoxGeometry(0.48, 0.42, 0.48);
                            const towerMat = new THREE.MeshLambertMaterial({{ color: 0x888888 }});
                            const tower = new THREE.Mesh(towerGeo, towerMat);
                            tower.position.set(0, 1.18, -0.36);
                            tower.castShadow = true;
                            group.add(tower);
                            const towerRoofGeo = new THREE.BoxGeometry(0.54, 0.07, 0.54);
                            const towerRoofMat = new THREE.MeshLambertMaterial({{ color: 0x555555 }});
                            const towerRoof = new THREE.Mesh(towerRoofGeo, towerRoofMat);
                            towerRoof.position.set(0, 1.44, -0.36);
                            group.add(towerRoof);
                            for (let bi = 0; bi < 4; bi++) {{
                                const angle = (bi / 4) * Math.PI * 2;
                                const bx = Math.sin(angle) * 0.8;
                                const bz = Math.cos(angle) * 0.8;
                                const bayGeo = new THREE.BoxGeometry(0.32, 0.4, 0.07);
                                const bayMat = new THREE.MeshLambertMaterial({{ color: 0x444444 }});
                                const bay = new THREE.Mesh(bayGeo, bayMat);
                                bay.position.set(bx, 0.32, bz);
                                bay.rotation.y = angle;
                                group.add(bay);
                            }}
                            const antGeo = new THREE.CylinderGeometry(0.028, 0.028, 0.4, 8);
                            const antMat = new THREE.MeshLambertMaterial({{ color: 0xaaaaaa }});
                            const ant = new THREE.Mesh(antGeo, antMat);
                            ant.position.set(0, 1.67, -0.36);
                            group.add(ant);
                            const lightGeo = new THREE.SphereGeometry(0.05, 8, 8);
                            const lightMat = new THREE.MeshBasicMaterial({{ color: 0xff0000 }});
                            const light = new THREE.Mesh(lightGeo, lightMat);
                            light.position.set(0, 1.88, -0.36);
                            group.add(light);
                        }} else {{
                            const geo = new THREE.BoxGeometry(1.2, 1.2, 1.2);
                            const mat = new THREE.MeshLambertMaterial({{ color: role.color }});
                            const mesh = new THREE.Mesh(geo, mat);
                            mesh.position.y = 0.6;
                            mesh.castShadow = true;
                            mesh.receiveShadow = true;
                            group.add(mesh);
                            group.userData.bodyMesh = mesh;
                        }}

                        group.position.set(...role.position, 0);
                        scene.add(group);
                        
                        // Add label
                        const canvas = document.createElement('canvas');
                        const context = canvas.getContext('2d');
                        canvas.width = 256;
                        canvas.height = 64;
                        context.fillStyle = "rgba(0,0,0,0.8)";
                        context.fillRect(0, 0, canvas.width, canvas.height);
                        context.fillStyle = "white";
                        context.font = "24px Arial";
                        context.textAlign = "center";
                        context.fillText(role.label, canvas.width/2, canvas.height/2 + 8);
                        
                        const texture = new THREE.CanvasTexture(canvas);
                        const spriteMaterial = new THREE.SpriteMaterial({{ map: texture }});
                        const sprite = new THREE.Sprite(spriteMaterial);
                        sprite.position.set(role.position[0], role.position[1] + 2, role.position[2]);
                        sprite.scale.set(2, 0.5, 1);
                        scene.add(sprite);
                        
                        // Create detail info panel
                        const infoPanel = createInfoPanel(role.name, role.label);
                        scene.add(infoPanel);
                        
                        agents[role.name] = {{
                            mesh: group,
                            sprite: sprite,
                            infoPanel: infoPanel,
                            data: {{}},
                            orderArrows: [],
                            shipmentArrows: []
                        }};
                    }});
                    // Draw supply chain connection lines between all nodes
                    const chainRoles = [
                        {{ name: 'customer', pos: [-20, 0.2, 0] }},
                        {{ name: 'retailer', pos: [-10, 0.2, 0] }},
                        {{ name: 'wholesaler', pos: [0, 0.2, 0] }},
                        {{ name: 'distributor', pos: [10, 0.2, 0] }},
                        {{ name: 'manufacturer', pos: [20, 0.2, 0] }}
                    ];
                    for (let ci = 0; ci < chainRoles.length - 1; ci++) {{
                        const from = new THREE.Vector3(...chainRoles[ci].pos);
                        const to = new THREE.Vector3(...chainRoles[ci + 1].pos);
                        const midPt = new THREE.Vector3().addVectors(from, to).multiplyScalar(0.5);

                        const lineGeo = new THREE.BufferGeometry().setFromPoints([from, to]);
                        const lineMat = new THREE.LineBasicMaterial({{ color: 0x888888, transparent: true, opacity: 0.35 }});
                        const line = new THREE.Line(lineGeo, lineMat);
                        scene.add(line);
                        line.userData = {{ from: chainRoles[ci].name, to: chainRoles[ci + 1].name }};
                        agents['line_' + chainRoles[ci].name] = line;

                        const arrGeo = new THREE.ConeGeometry(0.2, 0.7, 4);
                        const arrMat = new THREE.MeshLambertMaterial({{ color: 0x888888, transparent: true, opacity: 0.45 }});
                        const arr = new THREE.Mesh(arrGeo, arrMat);
                        arr.position.copy(midPt);
                        arr.position.y = 0.08;
                        arr.rotation.z = -Math.PI / 2;
                        scene.add(arr);
                        agents['arrow_' + chainRoles[ci].name] = arr;
                    }}
                    
                }}
                
                function createInfoPanel(agentName, agentLabel) {{
                    const canvas = document.createElement('canvas');
                    const context = canvas.getContext('2d');
                    canvas.width = 320;
                    canvas.height = 220;
                    
                    // Draw background
                    context.fillStyle = "rgba(0,0,0,0.8)";
                    context.fillRect(0, 0, canvas.width, canvas.height);

                    // Draw border
                    context.strokeStyle = "white";
                    context.lineWidth = 2;
                    context.strokeRect(2, 2, canvas.width-4, canvas.height-4);

                    // Draw title
                    context.fillStyle = "white";
                    context.font = "bold 16px Arial";
                    context.textAlign = "center";
                    context.fillText(agentLabel + ' Status', canvas.width/2, 25);

                    // Draw base data
                    context.font = "12px Arial";
                    context.textAlign = "left";
                    context.fillText('📦 Inventory: 0', 10, 50);
                    context.fillText('📋 Orders: 0', 10, 70);
                    context.fillText('🚚 In-Transit: 0', 10, 90);
                    context.fillText('❌ Backorder: 0', 10, 110);
                    context.fillText('📊 Holding: 0.00', 10, 130);
                    context.fillText('💸 Shortage Cost: 0.00', 160, 130);
                    context.fillText('💰 This Round: 0.00', 10, 150);
                    context.fillText('💸 Total: 0.00', 160, 150);
                    
                    // Remove overreaction indicator display
                    
                    const texture = new THREE.CanvasTexture(canvas);
                    const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
                    const sprite = new THREE.Sprite(spriteMaterial);
                    
                    // Adjust info panel position based on role
                    const positions = {
                        'customer': [-15, -4, 0],
                        'retailer': [-7.5, -4, 0],
                        'wholesaler': [0, -4, 0],
                        'distributor': [7.5, -4, 0],
                        'manufacturer': [15, -4, 0]
                    };
                    
                    if (positions[agentName]) {{
                        sprite.position.set(...positions[agentName]);
                    }}
                    sprite.scale.set(6, 3, 1);
                    
                    return sprite;
                }}
                
                function updateAgentInfo(agentName, data) {{
                    if (agents[[agentName]] == null || agents[[agentName]].infoPanel == null) return;
                    
                    const canvas = document.createElement('canvas');
                    const context = canvas.getContext('2d');
                    canvas.width = 320;
                    canvas.height = 220;
                    
                    // Draw background
                    context.fillStyle = 'rgba(0,0,0,0.8)';
                    context.fillRect(0, 0, canvas.width, canvas.height);

                    // Draw border
                    context.strokeStyle = 'white';
                    context.lineWidth = 2;
                    context.strokeRect(2, 2, canvas.width-4, canvas.height-4);

                    // Draw title
                    context.fillStyle = 'white';
                    context.font = 'bold 16px Arial';
                    context.textAlign = 'center';
                    const labels = {
                        'customer': 'Customer',
                        'retailer': 'Retailer',
                        'wholesaler': 'Wholesaler',
                        'distributor': 'Distributor',
                        'manufacturer': 'Manufacturer'
                    };
                    context.fillText((labels[agentName] || agentName) + ' Status', canvas.width/2, 25);

                    // Draw base data
                    context.font = '12px Arial';
                    context.textAlign = 'left';
                    
                    const inventory = data.inventory || 0;
                    const order = data.current_order || 0;
                    const intransit = Array.isArray(data.shipment_pipeline) ? 
                        data.shipment_pipeline.reduce((sum, val) => sum + val, 0) : 0;
                    const backorder = data.backorder || 0;
                    const roundCost = data.round_cost || 0;
                    const totalCost = data.total_cost || 0;
                    
                    // Calculate net inventory and set color based on unified status color standard
                    const netInventory = inventory - backorder;

                    if (netInventory > 0) {{
                        context.fillStyle = '#2196F3';
                    }} else if (netInventory === 0) {{
                        context.fillStyle = '#FFC107';
                    }} else {{
                        context.fillStyle = '#F44336';
                    }}
                    context.fillText('📦 Net Inventory: ' + netInventory, 10, 50);

                    context.fillStyle = 'white';
                    context.fillText('📋 Orders: ' + order, 10, 70);

                    context.fillStyle = intransit > 0 ? '#2196F3' : 'white';
                    context.fillText('🚚 In-Transit: ' + intransit, 10, 90);

                    context.fillStyle = backorder > 0 ? '#F44336' : 'white';
                    context.fillText('❌ Backorder: ' + backorder, 10, 110);

                    // Calculate cost breakdown (based on standard cost parameters)
                    const holdingCost = inventory * 0.5; // Inventory holding cost
                    const shortageCost = backorder * 1.0; // Shortage cost

                    context.fillStyle = '#4CAF50';
                    context.fillText('📊 Holding: ' + holdingCost.toFixed(2), 10, 130);
                    context.fillStyle = '#F44336';
                    context.fillText('💸 Shortage Cost: ' + shortageCost.toFixed(2), 160, 130);

                    context.fillStyle = '#FFD700';
                    context.fillText('💰 This Round: ' + roundCost.toFixed(2), 10, 150);
                    context.fillStyle = '#FFA500';
                    context.fillText('💸 Total: ' + totalCost.toFixed(2), 160, 150);
                    
                    // Remove overreaction indicator display
                    
                    // Update texture
                    const texture = new THREE.CanvasTexture(canvas);
                    agents[agentName].infoPanel.material.map = texture;
                    agents[agentName].infoPanel.material.needsUpdate = true;
                }}
                
                function createOrderArrow(fromAgent, toAgent, orderSize) {{
                    if (!agents[fromAgent] || !agents[toAgent]) return;
                    
                    const fromPos = agents[fromAgent].mesh.position;
                    const toPos = agents[toAgent].mesh.position;
                    
                    // Create arrow geometry
                    const direction = new THREE.Vector3().subVectors(toPos, fromPos).normalize();
                    const arrowHelper = new THREE.ArrowHelper(direction, fromPos,
                        fromPos.distanceTo(toPos) * 0.8, 0xff4444, 2, 1);

                    // Add order quantity label
                    const canvas = document.createElement('canvas');
                    const context = canvas.getContext('2d');
                    canvas.width = 128;
                    canvas.height = 32;
                    context.fillStyle = "rgba(255,68,68,0.9)";
                    context.fillRect(0, 0, canvas.width, canvas.height);
                    context.fillStyle = "white";
                    context.font = "bold 14px Arial";
                    context.textAlign = "center";
                    context.fillText('Order: ' + orderSize, canvas.width/2, canvas.height/2 + 5);
                    
                    const texture = new THREE.CanvasTexture(canvas);
                    const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
                    const sprite = new THREE.Sprite(spriteMaterial);
                    
                    const midPoint = new THREE.Vector3().addVectors(fromPos, toPos).multiplyScalar(0.5);
                    sprite.position.copy(midPoint);
                    sprite.position.y += 2;
                    sprite.scale.set(3, 0.8, 1);
                    
                    scene.add(arrowHelper);
                    scene.add(sprite);
                    
                    // Animation effect
                    let opacity = 1.0;
                    const fadeOut = () => {{
                        opacity -= 0.02;
                        if (opacity > 0) {{
                            arrowHelper.line.material.opacity = opacity;
                            arrowHelper.cone.material.opacity = opacity;
                            sprite.material.opacity = opacity;
                            requestAnimationFrame(fadeOut);
                        }} else {{
                            scene.remove(arrowHelper);
                            scene.remove(sprite);
                        }}
                    }};
                    
                    setTimeout(fadeOut, 2000);
                }}
                
                function createShipmentAnimation(fromAgent, toAgent, shipmentSize) {{
                    if (!agents[fromAgent] || !agents[toAgent]) return;
                    
                    const fromPos = agents[fromAgent].mesh.position.clone();
                    const toPos = agents[toAgent].mesh.position.clone();
                    
                    // Create shipment package
                    const packageGeometry = new THREE.BoxGeometry(0.5, 0.5, 0.5);
                    const packageMaterial = new THREE.MeshLambertMaterial({{ color: 0x8B4513 }});
                    const packageMesh = new THREE.Mesh(packageGeometry, packageMaterial);
                    packageMesh.position.copy(fromPos);
                    packageMesh.position.y += 2;
                    scene.add(packageMesh);
                    
                    // Add shipment label
                    const canvas = document.createElement('canvas');
                    const context = canvas.getContext('2d');
                    canvas.width = 128;
                    canvas.height = 32;
                    context.fillStyle = "rgba(139,69,19,0.9)";
                    context.fillRect(0, 0, canvas.width, canvas.height);
                    context.fillStyle = "white";
                    context.font = "bold 12px Arial";
                    context.textAlign = "center";
                    context.fillText('Shipment: ' + shipmentSize, canvas.width/2, canvas.height/2 + 4);
                    
                    const texture = new THREE.CanvasTexture(canvas);
                    const spriteMaterial = new THREE.SpriteMaterial({ map: texture });
                    const sprite = new THREE.Sprite(spriteMaterial);
                    sprite.position.copy(packageMesh.position);
                    sprite.position.y += 1;
                    sprite.scale.set(2, 0.5, 1);
                    scene.add(sprite);
                    
                    // Animate movement
                    const startTime = Date.now();
                    const duration = 3000; // 3 seconds move time

                    const animateShipment = () => {{
                        const elapsed = Date.now() - startTime;
                        const progress = Math.min(elapsed / duration, 1);

                        // Use easing function
                        const easeProgress = 1 - Math.pow(1 - progress, 3);

                        const currentPos = new THREE.Vector3().lerpVectors(fromPos, toPos, easeProgress);
                        currentPos.y += 2 + Math.sin(progress * Math.PI) * 1; // Parabolic trajectory

                        packageMesh.position.copy(currentPos);
                        sprite.position.copy(currentPos);
                        sprite.position.y += 1;

                        // Rotation effect
                        packageMesh.rotation.x += 0.1;
                        packageMesh.rotation.z += 0.05;

                        if (progress < 1) {{
                            requestAnimationFrame(animateShipment);
                        }} else {{
                            // Remove after reaching target
                            scene.remove(packageMesh);
                            scene.remove(sprite);
                        }}
                    }};
                    
                    animateShipment();
                }}
                
                function updateAgentNodes(agentsData, marketDemand) {{
                    Object.keys(agentsData).forEach(agentName => {{
                        const data = agentsData[agentName];
                        updateAgentInfo(agentName, data);
                        
                        // Update node color based on inventory status - unified three-state colors
                        if (agents[agentName] && agents[agentName].mesh) {{
                            const inventory = data.inventory || 0;
                            const backorder = data.backorder || 0;
                            const netInventory = inventory - backorder;
                            let color;
                            if (netInventory > 0) {{
                                color = 0x2196F3; // Blue - positive inventory
                            }} else if (netInventory === 0) {{
                                color = 0xFFC107; // Yellow - zero inventory
                            }} else {{
                                color = 0xF44336; // Red - negative inventory (backorder)
                            }}
                            const nodeGroup = agents[agentName].mesh;
                            if (nodeGroup.userData.bodyMesh) {{
                                nodeGroup.userData.bodyMesh.material.color.setHex(color);
                            }}
                            nodeGroup.children.forEach(child => {{
                                if (child.material && child.material.color) {{
                                    child.material.color.setHex(color);
                                    child.material.needsUpdate = true;
                                }}
                            }});
                        }}
                    }});
                    
                }}
                
                function animate() {{
                    requestAnimationFrame(animate);
                    controls.update();
                    renderer.render(scene, camera);
                }}
                
                // Window resize
                window.addEventListener('resize', () => {{
                    const container = document.getElementById('container');
                    const isFullscreen = container.classList.contains('fullscreen');
                    
                    if (isFullscreen) {{
                        camera.aspect = window.innerWidth / window.innerHeight;
                        camera.updateProjectionMatrix();
                        renderer.setSize(window.innerWidth, window.innerHeight);
                    }} else {{
                        camera.aspect = window.innerWidth / 1000;
                        camera.updateProjectionMatrix();
                        renderer.setSize(window.innerWidth, 1000);
                    }}
                }});
                
                // Fullscreen function
                function toggleFullscreen() {{
                    const container = document.getElementById('container');
                    const fullscreenBtn = document.getElementById('fullscreen-btn');
                    const exitBtn = document.getElementById('exit-fullscreen-btn');
                    
                    container.classList.add('fullscreen');
                    fullscreenBtn.style.display = 'none';
                    exitBtn.style.display = 'block';
                    
                    // Adjust renderer size
                    camera.aspect = window.innerWidth / window.innerHeight;
                    camera.updateProjectionMatrix();
                    renderer.setSize(window.innerWidth, window.innerHeight);

                    // Add ESC key to exit fullscreen
                    document.addEventListener('keydown', handleEscKey);
                }}
                
                function exitFullscreen() {{
                    const container = document.getElementById('container');
                    const fullscreenBtn = document.getElementById('fullscreen-btn');
                    const exitBtn = document.getElementById('exit-fullscreen-btn');
                    
                    container.classList.remove('fullscreen');
                    fullscreenBtn.style.display = 'block';
                    exitBtn.style.display = 'none';
                    
                    // Restore original size
                    camera.aspect = window.innerWidth / 1000;
                    camera.updateProjectionMatrix();
                    renderer.setSize(window.innerWidth, 1000);

                    // Remove ESC key listener
                    document.removeEventListener('keydown', handleEscKey);
                }}
                
                function handleEscKey(event) {{
                    if (event.key === "Escape") {{
                        exitFullscreen();
                    }}
                }}
                
                // Dynamically update agent node data
                function updateAgentNodes(agentsData) {{
                    Object.keys(agentsData).forEach(agentName => {{
                        const data = agentsData[agentName];
                        if (agents[agentName] && agents[agentName].mesh) {{
                            const inventory = data.inventory || 0;
                            let color;
                            if (inventory > 10) {{
                                color = 0x4CAF50; // Green - sufficient inventory
                            }} else if (inventory > 5) {{
                                color = 0xFFC107; // Yellow - moderate inventory
                            }} else if (inventory > 0) {{
                                color = 0xFF9800; // Orange - low inventory
                            }} else {{
                                color = 0xF44336; // Red - backorder
                            }}
                            const nodeGroup = agents[agentName].mesh;
                            if (nodeGroup.userData.bodyMesh) {{
                                nodeGroup.userData.bodyMesh.material.color.setHex(color);
                            }}
                            nodeGroup.children.forEach(child => {{
                                if (child.material && child.material.color) {{
                                    child.material.color.setHex(color);
                                    child.material.needsUpdate = true;
                                }}
                            }});
                        }}
                    }});
                }}
                
                // Create order arrow animation
                function createOrderArrow(fromAgent, toAgent, orderSize) {{
                    if (agents[fromAgent] && agents[toAgent]) {{
                        const fromPos = agents[fromAgent].mesh.position;
                        const toPos = agents[toAgent].mesh.position;

                        // Create arrow geometry
                        const arrowGeometry = new THREE.ConeGeometry(0.2, 1, 8);
                        const arrowMaterial = new THREE.MeshBasicMaterial({{color: 0x00ff00}});
                        const arrow = new THREE.Mesh(arrowGeometry, arrowMaterial);

                        // Set arrow position and direction
                        arrow.position.copy(fromPos);
                        const direction = new THREE.Vector3().subVectors(toPos, fromPos).normalize();
                        arrow.lookAt(toPos);

                        scene.add(arrow);

                        // Animate arrow movement
                        const startTime = Date.now();
                        const duration = 2000; // 2 seconds
                        
                        function animateArrow() {{
                            const elapsed = Date.now() - startTime;
                            const progress = Math.min(elapsed / duration, 1);
                            
                            arrow.position.lerpVectors(fromPos, toPos, progress);
                            
                            if (progress < 1) {{
                                requestAnimationFrame(animateArrow);
                            }} else {{
                                scene.remove(arrow);
                            }}
                        }}
                        
                        animateArrow();
                    }}
                }}
                
                // Create shipment animation
                function createShipmentAnimation(fromAgent, toAgent, shipmentSize) {{
                    if (agents[[fromAgent]] && agents[[toAgent]]) {{
                        const fromPos = agents[[fromAgent]].mesh.position;
                        const toPos = agents[[toAgent]].mesh.position;

                        // Create shipment geometry
                        const boxGeometry = new THREE.BoxGeometry(0.5, 0.5, 0.5);
                        const boxMaterial = new THREE.MeshBasicMaterial({{color: 0x0066cc}});
                        const box = new THREE.Mesh(boxGeometry, boxMaterial);

                        box.position.copy(fromPos);
                        scene.add(box);

                        // Animate shipment movement
                        const startTime = Date.now();
                        const duration = 3000; // 3 seconds
                        
                        function animateShipment() {{
                            const elapsed = Date.now() - startTime;
                            const progress = Math.min(elapsed / duration, 1);
                            
                            box.position.lerpVectors(fromPos, toPos, progress);
                            
                            if (progress < 1) {{
                                requestAnimationFrame(animateShipment);
                            }} else {{
                                scene.remove(box);
                            }}
                        }}
                        
                        animateShipment();
                    }}
                }}
                
                // Initialize scene
                init();

            </script>
        </body>
        </html>
        """
        
    def _create_updated_scene(self, round_data: Dict[str, Any]) -> str:
        """Create updated 3D scene HTML"""
        # Extract key data
        current_round = round_data.get('round', 0)
        agents_data = round_data.get('agents', {})
        market_demand = round_data.get('market_demand', 0)

        # Calculate totals
        total_inventory = sum(agent.get('inventory', 0) for agent in agents_data.values())
        total_intransit = sum(sum(agent.get('shipment_pipeline', [])) for agent in agents_data.values())
        total_backorder = sum(agent.get('backorder', 0) for agent in agents_data.values())

        # Calculate total cost
        total_cost = 0.0
        for agent_name, agent_data in agents_data.items():
            inventory = agent_data.get('inventory', 0)
            backorder = agent_data.get('backorder', 0)
            # Inventory holding cost (assume per-unit holding cost is 0.5)
            holding_cost = max(0, inventory) * 0.5
            # Shortage cost (assume per-unit shortage cost is 1.0)
            shortage_cost = max(0, backorder) * 1.0
            total_cost += holding_cost + shortage_cost

        # Generate updated HTML (including data update JavaScript)
        # Detect order and shipment animations
        animation_script = ""
        if hasattr(self, 'previous_data') and self.previous_data:
            prev_agents = self.previous_data.get('agents', {})
            curr_agents = agents_data

            # Detect new orders (order quantity increased)
            for agent_name in curr_agents:
                if agent_name in prev_agents:
                    prev_order = prev_agents[agent_name].get('order', 0)
                    curr_order = curr_agents[agent_name].get('order', 0)
                    if curr_order > prev_order:
                        # Determine order flow direction
                        upstream_map = {
                            'retailer': 'wholesaler',
                            'wholesaler': 'distributor',
                            'distributor': 'manufacturer'
                        }
                        if agent_name in upstream_map:
                            animation_script += f"createOrderArrow('{agent_name}', '{upstream_map[agent_name]}', {curr_order});\n"

                    # Detect shipments (in-transit goods increased)
                    prev_intransit = sum(prev_agents[agent_name].get('shipment_pipeline', []))
                    curr_intransit = sum(curr_agents[agent_name].get('shipment_pipeline', []))
                    if curr_intransit > prev_intransit:
                        # Determine shipment flow direction
                        downstream_map = {
                            'manufacturer': 'distributor',
                            'distributor': 'wholesaler',
                            'wholesaler': 'retailer'
                        }
                        if agent_name in downstream_map:
                            shipment_size = curr_intransit - prev_intransit
                            animation_script += f"createShipmentAnimation('{agent_name}', '{downstream_map[agent_name]}', {shipment_size});\n"

        # Save current data for next comparison
        self.previous_data = round_data.copy()

        update_script = f"""
        <script>
            // Update info panel
            document.getElementById('current-round').textContent = '{current_round}';
            document.getElementById('total-inventory').textContent = '{total_inventory}';
            document.getElementById('total-intransit').textContent = '{total_intransit}';
            document.getElementById('total-backorder').textContent = '{total_backorder}';
            document.getElementById('market-demand').textContent = '{market_demand}';
            document.getElementById('total-cost').textContent = '{total_cost:.2f}';
            document.getElementById('status').innerHTML = '🔄 Round {current_round} Running...';

            // Update 3D nodes
            if (typeof updateAgentNodes === 'function') {{
                updateAgentNodes({json.dumps(agents_data)}, {market_demand});
            }}

            // Execute animations
            {animation_script}
        </script>
        """

        # Return update script HTML
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>3D Scene Update</title>
        </head>
        <body>
            {update_script}
        </body>
        </html>
        """
        
    def create_streamlit_component(self):
        """Create Streamlit component interface"""
        col1, col2 = st.columns([3, 1])

        with col2:
            st.markdown("### 🎮 Real-Time 3D Controls")

            # Toggle control
            enable_3d = st.checkbox("🏢 Enable Real-Time 3D Display", value=False, key="enable_realtime_3d")

            if enable_3d:
                st.success("✅ Real-time 3D display enabled")
                if st.button("🔄 Reset View", key="reset_3d_view"):
                    st.rerun()
            else:
                st.info("ℹ️ Real-time 3D display disabled")

            # Display status
            st.markdown("#### 📊 Current Status")
            if hasattr(st.session_state, 'realtime_3d_round'):
                st.metric("Current Round", st.session_state.realtime_3d_round)
            else:
                st.metric("Current Round", "Not Started")

        with col1:
            if enable_3d:
                # Create 3D display container
                container = st.empty()
                if not hasattr(st.session_state, 'realtime_3d_viz'):
                    st.session_state.realtime_3d_viz = RealtimeSupplyChain3D()

                st.session_state.realtime_3d_viz.initialize_display(container)
                return st.session_state.realtime_3d_viz
            else:
                st.info("🎯 Enable real-time 3D display to view dynamic supply chain visualization")
                return None
                
    @staticmethod
    def extract_round_data(agents: Dict, round_num: int, market_demand: int = 0) -> Dict[str, Any]:
        """Extract round information from agent data"""
        round_data = {
            'round': round_num,
            'market_demand': market_demand,
            'agents': {}
        }
        
        for role, agent in agents.items():
            if hasattr(agent, 'inventory') and hasattr(agent, 'backorder'):
                round_data['agents'][role] = {
                    'inventory': getattr(agent, 'inventory', 0),
                    'backorder': getattr(agent, 'backorder', 0),
                    'shipment_pipeline': getattr(agent, 'shipment_pipeline', []),
                    'order_history': getattr(agent, 'order_history', []),
                    'demand_history': getattr(agent, 'demand_history', [])
                }
                
        return round_data
        
    def _create_update_script(self, round_data: Dict[str, Any]) -> str:
        """Create update script"""
        current_round = round_data.get('round', 0)
        agents_data = round_data.get('agents', {})
        market_demand = round_data.get('market_demand', 0)

        # Calculate totals
        total_inventory = sum(agent.get('inventory', 0) for agent in agents_data.values())
        total_intransit = sum(sum(agent.get('shipment_pipeline', [])) for agent in agents_data.values())
        total_backorder = sum(agent.get('backorder', 0) for agent in agents_data.values())

        # Calculate total cost
        total_cost = 0.0
        for agent_name, agent_data in agents_data.items():
            inventory = agent_data.get('inventory', 0)
            backorder = agent_data.get('backorder', 0)
            # Inventory holding cost (assume per-unit holding cost is 0.5)
            holding_cost = max(0, inventory) * 0.5
            # Shortage cost (assume per-unit shortage cost is 1.0)
            shortage_cost = max(0, backorder) * 1.0
            total_cost += holding_cost + shortage_cost

        # Generate animation script
        animation_script = ""
        if hasattr(self, 'previous_data') and self.previous_data:
            prev_agents = self.previous_data.get('agents', {})
            curr_agents = agents_data

            # Detect new orders
            for agent_name in curr_agents:
                if agent_name in prev_agents:
                    prev_order = prev_agents[agent_name].get('current_order', 0)
                    curr_order = curr_agents[agent_name].get('current_order', 0)
                    if curr_order > prev_order:
                        upstream_map = {
                            'retailer': 'wholesaler',
                            'wholesaler': 'distributor',
                            'distributor': 'manufacturer'
                        }
                        if agent_name in upstream_map:
                            animation_script += f"createOrderArrow('{agent_name}', '{upstream_map[agent_name]}', {curr_order});\n"

                    # Detect shipments
                    prev_intransit = sum(prev_agents[agent_name].get('shipment_pipeline', []))
                    curr_intransit = sum(curr_agents[agent_name].get('shipment_pipeline', []))
                    if curr_intransit > prev_intransit:
                        downstream_map = {
                            'manufacturer': 'distributor',
                            'distributor': 'wholesaler',
                            'wholesaler': 'retailer'
                        }
                        if agent_name in downstream_map:
                            shipment_size = curr_intransit - prev_intransit
                            animation_script += f"createShipmentAnimation('{agent_name}', '{downstream_map[agent_name]}', {shipment_size});\n"

        # Remove overreaction indicator update script generation
        overreaction_updates = ""

        # Save current data for next comparison
        self.previous_data = round_data.copy()

        return f"""
            // Update info panel
            if (document.getElementById('current-round')) {{
                document.getElementById('current-round').textContent = '{current_round}';
                document.getElementById('total-inventory').textContent = '{total_inventory}';
                document.getElementById('total-intransit').textContent = '{total_intransit}';
                document.getElementById('total-backorder').textContent = '{total_backorder}';
                document.getElementById('market-demand').textContent = '{market_demand}';
                document.getElementById('total-cost').textContent = '{total_cost:.2f}';
                document.getElementById('status').innerHTML = '🔄 Round {current_round} Running...';
            }}

            // Update 3D nodes
            if (typeof updateAgentNodes === 'function') {{
                updateAgentNodes({json.dumps(agents_data)});
            }}

            // Remove overreaction indicator update

            // Execute animations
            {animation_script}
        """