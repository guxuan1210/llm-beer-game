"""3D Supply Chain Dynamic Visualization Module

Provides Three.js-based 3D dynamic supply chain visualization functionality.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import streamlit as st
import streamlit.components.v1 as components


class SupplyChain3DVisualizer:
    """3D Supply Chain Visualizer"""
    
    def __init__(self):
        self.template_path = Path(__file__).parent / "templates"
        self.template_path.mkdir(exist_ok=True)
        
    def create_3d_visualization(self, simulation_result, width: int = 1200, height: int = 800) -> str:
        """Create 3D supply chain visualization

        Args:
            simulation_result: Simulation result data
            width: Visualization width
            height: Visualization height

        Returns:
            HTML string
        """
        # Extract simulation data
        visualization_data = self._extract_visualization_data(simulation_result)

        # Generate HTML template
        html_content = self._generate_html_template(visualization_data, width, height)
        
        return html_content
    
    def _extract_visualization_data(self, simulation_result) -> Dict[str, Any]:
        """Extract data required for visualization"""
        data = {
            "rounds": [],
            "agents": {
                "retailer": {"name": "Retailer", "position": [0, 0, 0], "color": "#FF6B6B"},
                "wholesaler": {"name": "Wholesaler", "position": [10, 0, 0], "color": "#4ECDC4"},
                "distributor": {"name": "Distributor", "position": [20, 0, 0], "color": "#45B7D1"},
                "manufacturer": {"name": "Manufacturer", "position": [30, 0, 0], "color": "#96CEB4"}
            },
            "customer": {"name": "Customer", "position": [-10, 0, 0], "color": "#FECA57"}
        }

        # Extract data for each round
        for round_data in simulation_result.round_history:
            round_info = {
                "round": round_data["round"],
                "demand": round_data.get("demand", 0),
                "customer_demand": round_data.get("customer_demand", 0),  # Add market demand field
                "agents": {},
                "orders_flow": round_data.get("orders_flow", []),
                "shipments": [],
                "coordinator_suggestions": {},  # Disabled
            }
            
            # Extract agent states
            for agent_key in ["retailer", "wholesaler", "distributor", "manufacturer"]:
                if agent_key in round_data["agents"]:
                    agent_data = round_data["agents"][agent_key]
                    round_info["agents"][agent_key] = {
                        "inventory": agent_data["end_state"].get("inventory", 0),
                        "backorder": agent_data["end_state"].get("backorder", 0),
                        "in_transit": agent_data["end_state"].get("total_in_transit", 0),
                        "incoming_shipment": agent_data["end_state"].get("incoming_shipment", 0),
                        "order_placed": agent_data.get("order_placed", 0),
                        "demand_received": agent_data.get("demand_received", 0),
                        "shipment_pipeline": agent_data["end_state"].get("shipment_pipeline", []),
                        "coordinator_suggestion": ""  # Disabled
                    }
            
            # Generate shipment flow data
            for order in round_data.get("orders_flow", []):
                if "from" in order and "to" in order:
                    round_info["shipments"].append({
                        "from": order["from"],
                        "to": order["to"],
                        "quantity": order.get("quantity", 0),
                        "type": "order"
                    })
            
            data["rounds"].append(round_info)
        
        # Add debug info to verify demand data extraction
        print("Extracted demand data for first 3 rounds:")
        for i, round_info in enumerate(data["rounds"][:3]):
            print(f"Round {i+1}: customer_demand={round_info.get('customer_demand', 0)}, demand={round_info.get('demand', 0)}")
        
        return data
    
    def _generate_html_template(self, data: Dict[str, Any], width: int, height: int) -> str:
        """Generate HTML template"""
        data_json = json.dumps(data, ensure_ascii=False, indent=2)
        
        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>3D Supply Chain Dynamic Visualization</title>
    <!-- Three.js dynamic loader with multi-CDN fallback -->
    <script>
        (function() {{
            var THREE_URLS = [
                'https://unpkg.com/three@0.128.0/build/three.min.js',
                'https://cdn.jsdelivr.net/npm/three@0.128.0/build/three.min.js',
                'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js'
            ];
            var ORBIT_URLS = [
                'https://unpkg.com/three@0.128.0/examples/js/controls/OrbitControls.js',
                'https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js'
            ];
            window._threeReady = false;
            window._orbitReady = false;
            window._initCalled = false;

            function loadJs(urls, onAllFail) {{
                var idx = 0;
                function tryNext() {{
                    if (idx >= urls.length) {{ onAllFail(); return; }}
                    var s = document.createElement('script');
                    s.src = urls[idx++];
                    s.onload = function() {{ /* loaded */ }};
                    s.onerror = tryNext;
                    document.head.appendChild(s);
                }}
                tryNext();
            }}

            function checkReady() {{
                if (window._initCalled) return;
                if (typeof THREE !== 'undefined' && typeof THREE.OrbitControls !== 'undefined') {{
                    window._initCalled = true;
                    var overlay = document.getElementById('loading-overlay');
                    if (overlay) overlay.style.display = 'none';
                    if (typeof window._bootScene === 'function') window._bootScene();
                }} else if (typeof THREE !== 'undefined') {{
                    setTimeout(checkReady, 100);
                }}
            }}

            window._bootScene = null;

            loadJs(THREE_URLS, function() {{
                var overlay = document.getElementById('loading-overlay');
                if (overlay) overlay.innerHTML = '<div style="color:#f44;font-size:18px;">&#10060; Failed to load 3D engine<br><small>Check network and refresh</small></div>';
            }});

            setTimeout(function() {{
                loadJs(ORBIT_URLS, function() {{
                    var overlay = document.getElementById('loading-overlay');
                    if (overlay) overlay.innerHTML = '<div style="color:#f44;font-size:18px;">&#10060; Failed to load 3D controls<br><small>Check network and refresh</small></div>';
                }});
                var poll = setInterval(function() {{
                    checkReady();
                    if (window._initCalled) clearInterval(poll);
                }}, 150);
            }}, 50);
        }})();
    </script>
    <style>
        body {{
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Arial', sans-serif;
            overflow: hidden;
        }}
        
        #container {{
            width: {width}px;
            height: {height}px;
            position: relative;
            margin: 0 auto;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        }}

        #loading-overlay {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            z-index: 200;
            background: rgba(0,0,0,0.85);
            color: #fff;
            padding: 30px 40px;
            border-radius: 12px;
            text-align: center;
            font-family: Arial, sans-serif;
            font-size: 16px;
        }}

        #loading-overlay .spinner {{
            width: 40px;
            height: 40px;
            margin: 0 auto 15px;
            border: 4px solid rgba(255,255,255,0.3);
            border-top: 4px solid #fff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}

        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}



        #controls {{
            position: absolute;
            top: 10px;
            left: 10px;
            z-index: 100;
            background: rgba(255,255,255,0.92);
            padding: 8px 10px;
            border-radius: 6px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
            font-size: 11px;
            max-width: 200px;
        }}
        #controls.collapsed .control-group {{
            display: none;
        }}
        #controls.collapsed {{
            padding: 4px 10px;
        }}

        #info {{
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 100;
            background: rgba(255,255,255,0.92);
            padding: 8px 10px;
            border-radius: 6px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
            min-width: 180px;
            max-width: 230px;
            max-height: calc(100% - 20px);
            overflow-y: auto;
            font-size: 11px;
        }}
        #info.collapsed .info-item,
        #info.collapsed .agent-data,
        #info.collapsed h4 {{
            display: none;
        }}
        #info.collapsed {{
            padding: 4px 10px;
        }}
        
        .agent-data {{
            margin-top: 6px;
            border-top: 1px solid #eee;
            padding-top: 6px;
        }}

        .agent-section {{
            margin-bottom: 6px;
            padding: 5px 6px;
            border-radius: 4px;
            background-color: rgba(240,240,240,0.5);
        }}

        .panel-header {{
            cursor: pointer;
            user-select: none;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        .panel-header .toggle-icon {{
            font-size: 10px;
            transition: transform 0.2s;
            color: #888;
        }}
        .collapsed .panel-header .toggle-icon {{
            transform: rotate(-90deg);
        }}

        .control-group {{
            margin-bottom: 6px;
        }}

        .control-group label {{
            display: block;
            margin-bottom: 3px;
            font-weight: bold;
            color: #333;
            font-size: 11px;
        }}

        .control-group input, .control-group button {{
            width: 100%;
            padding: 4px 6px;
            border: 1px solid #ddd;
            border-radius: 4px;
            font-size: 11px;
        }}

        .control-group button {{
            background: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            transition: background 0.3s;
            padding: 5px 8px;
        }}

        .control-group button:hover {{
            background: #45a049;
        }}

        .control-group button:disabled {{
            background: #cccccc;
            cursor: not-allowed;
        }}

        #info h3 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 3px;
            font-size: 13px;
        }}
        #info h4 {{
            font-size: 11px;
            margin-top: 8px;
            margin-bottom: 4px;
        }}
        #info h5 {{
            font-size: 11px;
            margin: 3px 0;
        }}

        .info-item {{
            margin-bottom: 3px;
            font-size: 11px;
        }}

        .info-label {{
            font-weight: bold;
            color: #555;
        }}

        .info-value {{
            color: #333;
        }}

        .legend {{
            position: absolute;
            bottom: 10px;
            left: 10px;
            z-index: 100;
            background: rgba(255,255,255,0.92);
            padding: 8px 10px;
            border-radius: 6px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.2);
            font-size: 11px;
        }}
        .legend.collapsed .legend-item {{
            display: none;
        }}
        .legend.collapsed {{
            padding: 4px 10px;
        }}

        .legend-item {{
            display: flex;
            align-items: center;
            margin-bottom: 4px;
            font-size: 11px;
        }}

        .legend-color {{
            width: 14px;
            height: 14px;
            border-radius: 50%;
            margin-right: 8px;
        }}
    </style>
</head>
<body>
    <div id="container">
        <div id="loading-overlay">
            <div class="spinner"></div>
            Loading 3D engine...
        </div>
<div id="controls">
            <div class="panel-header" onclick="togglePanel('controls')">
                <span><b>🎮 Controls</b></span>
                <span class="toggle-icon">▼</span>
            </div>
            <div class="control-group">
                <label for="roundSlider">Round:</label>
                <input type="range" id="roundSlider" min="0" max="0" value="0">
                <span id="roundDisplay">Round 1</span>
            </div>
            <div class="control-group">
                <button id="playButton">▶️ Play</button>
            </div>
            <div class="control-group">
                <label for="speedSlider">Speed:</label>
                <input type="range" id="speedSlider" min="1" max="20" value="5">
                <span id="speedDisplay">5x</span>
            </div>
            <div class="control-group">
                <button id="resetButton">🔄 Reset</button>
            </div>
        </div>
        
        <div id="info">
            <div class="panel-header" onclick="togglePanel('info')">
                <span><b>📊 Status</b></span>
                <span class="toggle-icon">▼</span>
            </div>
            <div class="info-item">
                <span class="info-label">Round:</span>
                <span class="info-value" id="currentRound">1</span>
            </div>
            <div class="info-item">
                <span class="info-label">Market Demand:</span>
                <span class="info-value" id="currentDemand">0</span>
            </div>
            <div class="info-item">
                <span class="info-label">Total Inventory:</span>
                <span class="info-value" id="totalInventory">0</span>
            </div>
            <div class="info-item">
                <span class="info-label">Total In-Transit:</span>
                <span class="info-value" id="totalInTransit">0</span>
            </div>
            <div class="info-item">
                <span class="info-label">Total Backorder:</span>
                <span class="info-value" id="totalBackorder">0</span>
            </div>

            <h4>🏢 Participant Data</h4>
            <div class="agent-data">
                <div class="agent-section">
                    <h5 style="margin: 5px 0; color: #FF6B6B;">Retailer</h5>
                    <div class="info-item">
                        <span class="info-label">Demand:</span>
                        <span class="info-value" id="retailerDemand">0</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Inventory:</span>
                        <span class="info-value" id="retailerInventory">0</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Backorder:</span>
                        <span class="info-value" id="retailerBackorder">0</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Orders:</span>
                        <span class="info-value" id="retailerOrder">0</span>
                    </div>
                </div>
                
                <div class="agent-section">
                    <h5 style="margin: 3px 0; color: #4ECDC4;">Wholesaler</h5>
                    <div class="info-item">
                        <span class="info-label">Demand:</span>
                        <span class="info-value" id="wholesalerDemand">0</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Inventory:</span>
                        <span class="info-value" id="wholesalerInventory">0</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Backorder:</span>
                        <span class="info-value" id="wholesalerBackorder">0</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Orders:</span>
                        <span class="info-value" id="wholesalerOrder">0</span>
                    </div>
                </div>

                <div class="agent-section">
                    <h5 style="margin: 3px 0; color: #45B7D1;">Distributor</h5>
                    <div class="info-item">
                        <span class="info-label">Demand:</span>
                        <span class="info-value" id="distributorDemand">0</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Inventory:</span>
                        <span class="info-value" id="distributorInventory">0</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Backorder:</span>
                        <span class="info-value" id="distributorBackorder">0</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Orders:</span>
                        <span class="info-value" id="distributorOrder">0</span>
                    </div>
                </div>

                <div class="agent-section">
                    <h5 style="margin: 3px 0; color: #96CEB4;">Manufacturer</h5>
                    <div class="info-item">
                        <span class="info-label">Demand:</span>
                        <span class="info-value" id="manufacturerDemand">0</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Inventory:</span>
                        <span class="info-value" id="manufacturerInventory">0</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Backorder:</span>
                        <span class="info-value" id="manufacturerBackorder">0</span>
                    </div>
                    <div class="info-item">
                        <span class="info-label">Orders:</span>
                        <span class="info-value" id="manufacturerOrder">0</span>
                    </div>
                </div>
            </div>
            </div>

        <div class="legend">
            <div class="panel-header" onclick="togglePanel('legend')" style="margin-bottom:4px;">
                <span><b>🏢 Roles</b></span>
                <span class="toggle-icon">▼</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #FECA57;"></div>
                <span>Customer</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #FF6B6B;"></div>
                <span>Retailer</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #4ECDC4;"></div>
                <span>Wholesaler</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #45B7D1;"></div>
                <span>Distributor</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #96CEB4;"></div>
                <span>Manufacturer</span>
            </div>
        </div>
    </div>
    
    <script>
        // Simulation data
        const simulationData = {data_json};

        // Add debug info to verify data is correctly passed
        console.log("Simulation data loaded:");
        console.log("Number of rounds: " + simulationData.rounds.length);
        console.log("First 3 rounds demand data:");
        for (let i = 0; i < Math.min(3, simulationData.rounds.length); i++) {{
            console.log("Round " + (i+1) + ": customer_demand=" + simulationData.rounds[i].customer_demand + ", demand=" + simulationData.rounds[i].demand);
        }}
        
        // Three.js scene setup
        let scene, camera, renderer, controls;
        let agentMeshes = {{}};
        let orderFlows = [];
        let animationId;
        let isPlaying = false;
        let currentRoundIndex = 0;
        let playSpeed = 3;
        
        // Toggle panel collapse/expand
        function togglePanel(panelId) {{
            const panel = panelId === 'legend'
                ? document.querySelector('.legend')
                : document.getElementById(panelId);
            if (panel) {{
                panel.classList.toggle('collapsed');
            }}
        }}

        // Initialize 3D scene
        function initScene() {{
            // Create scene
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0xf0f0f0);
            
            // Create camera
            camera = new THREE.PerspectiveCamera(75, {width}/{height}, 0.1, 1000);
            camera.position.set(10, 14, 28);
            
            // Create renderer
            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setSize({width}, {height});
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            document.getElementById('container').appendChild(renderer.domElement);
            
            // Add controls
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            
            // Add lights
            const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(10, 10, 5);
            directionalLight.castShadow = true;
            directionalLight.shadow.mapSize.width = 2048;
            directionalLight.shadow.mapSize.height = 2048;
            scene.add(directionalLight);
            
            // Add ground
            const groundGeometry = new THREE.PlaneGeometry(75, 25);
            const groundMaterial = new THREE.MeshLambertMaterial({{ color: 0xffffff, transparent: true, opacity: 0.8 }});
            const ground = new THREE.Mesh(groundGeometry, groundMaterial);
            ground.rotation.x = -Math.PI / 2;
            ground.receiveShadow = true;
            scene.add(ground);

            // Add raised platform for all participants
            const platformGeo = new THREE.BoxGeometry(58, 0.15, 10);
            const platformMat = new THREE.MeshLambertMaterial({{ color: 0x444444 }});
            const platform = new THREE.Mesh(platformGeo, platformMat);
            platform.position.set(10, -0.08, 0);
            platform.receiveShadow = true;
            platform.castShadow = true;
            scene.add(platform);

            // Add platform border/edge highlight
            const edgeGeo = new THREE.EdgesGeometry(platformGeo);
            const edgeMat = new THREE.LineBasicMaterial({{ color: 0x666666 }});
            const edgeLine = new THREE.LineSegments(edgeGeo, edgeMat);
            edgeLine.position.copy(platform.position);
            scene.add(edgeLine);

            // Add direction markers on platform (arrows showing flow: Customer → Manufacturer)
            const flowLabels = ['Demand', 'Orders', 'Products'];
            for (let fi = 0; fi < 3; fi++) {{
                const lblCanvas = document.createElement('canvas');
                const lblCtx = lblCanvas.getContext('2d');
                lblCanvas.width = 256;
                lblCanvas.height = 32;
                lblCtx.fillStyle = 'rgba(255,255,255,0.7)';
                lblCtx.fillRect(0, 0, lblCanvas.width, lblCanvas.height);
                lblCtx.fillStyle = '#444444';
                lblCtx.font = '14px Arial';
                lblCtx.textAlign = 'center';
                lblCtx.fillText(flowLabels[fi], lblCanvas.width/2, lblCanvas.height/2 + 5);
                const lblTex = new THREE.CanvasTexture(lblCanvas);
                const lblMat = new THREE.SpriteMaterial({{ map: lblTex, transparent: true, opacity: 0.5 }});
                const lblSprite = new THREE.Sprite(lblMat);
                lblSprite.position.set(-8 + fi * 18, 0.05, 3.5);
                lblSprite.scale.set(3, 0.4, 1);
                scene.add(lblSprite);
            }}
            
            // Create supply chain nodes
            createSupplyChainNodes();

            // Initialize controls
            initControls();

            // Start render loop
            animate();

            // Show first round data
            updateVisualization(0);
        }}
        
        // Create supply chain nodes
        function createSupplyChainNodes() {{
            // Create customer node (3D person figure)
            const customerGroup = new THREE.Group();

            // Head
            const headGeo = new THREE.SphereGeometry(0.25, 16, 16);
            const headMat = new THREE.MeshLambertMaterial({{ color: 0xFFDBA4 }});
            const head = new THREE.Mesh(headGeo, headMat);
            head.position.y = 1.95;
            head.castShadow = true;
            customerGroup.add(head);

            // Body
            const bodyGeo = new THREE.CylinderGeometry(0.2, 0.28, 0.8, 8);
            const bodyMat = new THREE.MeshLambertMaterial({{ color: 0x4488CC }});
            const body = new THREE.Mesh(bodyGeo, bodyMat);
            body.position.y = 1.15;
            body.castShadow = true;
            customerGroup.add(body);
            customerGroup.userData.bodyMesh = body;

            // Left arm
            const leftArmGeo = new THREE.CylinderGeometry(0.07, 0.07, 0.6, 8);
            const leftArm = new THREE.Mesh(leftArmGeo, bodyMat);
            leftArm.position.set(-0.35, 1.3, 0);
            leftArm.rotation.z = 0.2;
            leftArm.castShadow = true;
            customerGroup.add(leftArm);

            // Right arm
            const rightArmGeo = new THREE.CylinderGeometry(0.07, 0.07, 0.6, 8);
            const rightArm = new THREE.Mesh(rightArmGeo, bodyMat);
            rightArm.position.set(0.35, 1.3, 0);
            rightArm.rotation.z = -0.2;
            rightArm.castShadow = true;
            customerGroup.add(rightArm);

            // Left leg
            const leftLegGeo = new THREE.CylinderGeometry(0.1, 0.1, 0.7, 8);
            const legsMat = new THREE.MeshLambertMaterial({{ color: 0x335577 }});
            const leftLeg = new THREE.Mesh(leftLegGeo, legsMat);
            leftLeg.position.set(-0.12, 0.4, 0);
            leftLeg.castShadow = true;
            customerGroup.add(leftLeg);

            // Right leg
            const rightLegGeo = new THREE.CylinderGeometry(0.1, 0.1, 0.7, 8);
            const rightLeg = new THREE.Mesh(rightLegGeo, legsMat);
            rightLeg.position.set(0.12, 0.4, 0);
            rightLeg.castShadow = true;
            customerGroup.add(rightLeg);

            customerGroup.position.set(-10, 0, 0);
            scene.add(customerGroup);
            agentMeshes['customer'] = customerGroup;
            
            // Create agent nodes (distinctive 3D shapes for each role)
            const agents = simulationData.agents;
            Object.keys(agents).forEach(agentKey => {{
                const agent = agents[agentKey];
                const colorHex = parseInt(agent.color.replace('#', '0x'));
                const group = new THREE.Group();

                if (agentKey === 'manufacturer') {{
                    // Factory: building with chimney and smoke
                    const bodyGeo = new THREE.BoxGeometry(1.3, 0.95, 1.3);
                    const bodyMat = new THREE.MeshLambertMaterial({{ color: colorHex }});
                    const body = new THREE.Mesh(bodyGeo, bodyMat);
                    body.position.y = 0.475;
                    body.castShadow = true;
                    group.add(body);
                    group.userData.bodyMesh = body;

                    const roofGeo = new THREE.ConeGeometry(0.75, 0.4, 4);
                    const roofMat = new THREE.MeshLambertMaterial({{ color: 0x555555 }});
                    const roof = new THREE.Mesh(roofGeo, roofMat);
                    roof.position.y = 1.15;
                    roof.rotation.y = Math.PI / 4;
                    roof.castShadow = true;
                    group.add(roof);

                    const chimneyGeo = new THREE.CylinderGeometry(0.11, 0.12, 0.7, 8);
                    const chimneyMat = new THREE.MeshLambertMaterial({{ color: 0x888888 }});
                    const chimney = new THREE.Mesh(chimneyGeo, chimneyMat);
                    chimney.position.set(0.35, 1.3, 0.24);
                    chimney.castShadow = true;
                    group.add(chimney);

                    for (let si = 0; si < 3; si++) {{
                        const sGeo = new THREE.SphereGeometry(0.07 + si * 0.035, 8, 8);
                        const sMat = new THREE.MeshLambertMaterial({{ color: 0xcccccc, transparent: true, opacity: 0.6 - si * 0.15 }});
                        const sMesh = new THREE.Mesh(sGeo, sMat);
                        sMesh.position.set(0.3 + si * 0.035, 1.55 + si * 0.15, 0.22);
                        group.add(sMesh);
                    }}
                }} else if (agentKey === 'retailer') {{
                    // Store/shop: building with awning, door, and sign
                    const bodyGeo = new THREE.BoxGeometry(1.3, 0.95, 1.15);
                    const bodyMat = new THREE.MeshLambertMaterial({{ color: colorHex }});
                    const body = new THREE.Mesh(bodyGeo, bodyMat);
                    body.position.y = 0.475;
                    body.castShadow = true;
                    group.add(body);
                    group.userData.bodyMesh = body;

                    const roofGeo = new THREE.ConeGeometry(0.8, 0.4, 4)
                    const roofMat = new THREE.MeshLambertMaterial({{ color: 0x8B4513 }});
                    const roof = new THREE.Mesh(roofGeo, roofMat);
                    roof.position.y = 1.15;
                    roof.rotation.y = Math.PI / 4;
                    roof.castShadow = true;
                    group.add(roof);

                    const awningGeo = new THREE.BoxGeometry(0.85, 0.06, 0.3);
                    const awningMat = new THREE.MeshLambertMaterial({{ color: 0x444444 }});
                    const awning = new THREE.Mesh(awningGeo, awningMat);
                    awning.position.set(0, 0.48, 0.72);
                    group.add(awning);

                    const doorGeo = new THREE.BoxGeometry(0.3, 0.42, 0.05);
                    const doorMat = new THREE.MeshLambertMaterial({{ color: 0x8B4513 }});
                    const door = new THREE.Mesh(doorGeo, doorMat);
                    door.position.set(0, 0.21, 0.52);
                    group.add(door);

                    const signGeo = new THREE.BoxGeometry(0.36, 0.18, 0.05);
                    const signMat = new THREE.MeshLambertMaterial({{ color: 0xFFD700 }});
                    const sign = new THREE.Mesh(signGeo, signMat);
                    sign.position.set(0, 0.66, 0.52);
                    group.add(sign);
                }} else if (agentKey === 'wholesaler') {{
                    // Warehouse: wide building with loading docks
                    const bodyGeo = new THREE.BoxGeometry(1.6, 0.9, 1.45);
                    const bodyMat = new THREE.MeshLambertMaterial({{ color: colorHex }});
                    const body = new THREE.Mesh(bodyGeo, bodyMat);
                    body.position.y = 0.45;
                    body.castShadow = true;
                    group.add(body);
                    group.userData.bodyMesh = body;

                    const roofGeo = new THREE.BoxGeometry(1.74, 0.08, 1.56);
                    const roofMat = new THREE.MeshLambertMaterial({{ color: 0x666666 }});
                    const roof = new THREE.Mesh(roofGeo, roofMat);
                    roof.position.y = 0.92;
                    roof.castShadow = true;
                    group.add(roof);

                    for (let di = -1; di <= 1; di++) {{
                        const dockGeo = new THREE.BoxGeometry(0.3, 0.22, 0.18);
                        const dockMat = new THREE.MeshLambertMaterial({{ color: 0x333333 }});
                        const dock = new THREE.Mesh(dockGeo, dockMat);
                        dock.position.set(di * 0.36, 0.15, 0.67);
                        group.add(dock);
                        const doorGeo = new THREE.BoxGeometry(0.25, 0.25, 0.05);
                        const doorMat = new THREE.MeshLambertMaterial({{ color: 0x777777 }});
                        const door = new THREE.Mesh(doorGeo, doorMat);
                        door.position.set(di * 0.36, 0.4, 0.66);
                        group.add(door);
                    }}
                    for (let vi = -1; vi <= 1; vi += 2) {{
                        const ventGeo = new THREE.CylinderGeometry(0.07, 0.07, 0.22, 8);
                        const ventMat = new THREE.MeshLambertMaterial({{ color: 0x999999 }});
                        const vent = new THREE.Mesh(ventGeo, ventMat);
                        vent.position.set(vi * 0.42, 0.96, 0.22);
                        group.add(vent);
                    }}
                }} else if (agentKey === 'distributor') {{
                    // Distribution center: building with office tower, loading bays, and antenna
                    const bodyGeo = new THREE.BoxGeometry(1.5, 0.9, 1.5);
                    const bodyMat = new THREE.MeshLambertMaterial({{ color: colorHex }});
                    const body = new THREE.Mesh(bodyGeo, bodyMat);
                    body.position.y = 0.45;
                    body.castShadow = true;
                    group.add(body);
                    group.userData.bodyMesh = body;

                    const roofGeo = new THREE.BoxGeometry(1.56, 0.08, 1.56);
                    const roofMat = new THREE.MeshLambertMaterial({{ color: 0x666666 }});
                    const roof = new THREE.Mesh(roofGeo, roofMat);
                    roof.position.y = 0.92;
                    roof.castShadow = true;
                    group.add(roof);

                    const towerGeo = new THREE.BoxGeometry(0.42, 0.36, 0.42);
                    const towerMat = new THREE.MeshLambertMaterial({{ color: 0x888888 }});
                    const tower = new THREE.Mesh(towerGeo, towerMat);
                    tower.position.set(0, 1.01, -0.3);
                    tower.castShadow = true;
                    group.add(tower);

                    const towerRoofGeo = new THREE.BoxGeometry(0.48, 0.06, 0.48);
                    const towerRoofMat = new THREE.MeshLambertMaterial({{ color: 0x555555 }});
                    const towerRoof = new THREE.Mesh(towerRoofGeo, towerRoofMat);
                    towerRoof.position.set(0, 1.22, -0.3);
                    group.add(towerRoof);

                    for (let bi = 0; bi < 4; bi++) {{
                        const angle = (bi / 4) * Math.PI * 2;
                        const bx = Math.sin(angle) * 0.7;
                        const bz = Math.cos(angle) * 0.7;
                        const bayGeo = new THREE.BoxGeometry(0.3, 0.36, 0.06);
                        const bayMat = new THREE.MeshLambertMaterial({{ color: 0x444444 }});
                        const bay = new THREE.Mesh(bayGeo, bayMat);
                        bay.position.set(bx, 0.28, bz);
                        bay.rotation.y = angle;
                        group.add(bay);
                    }}

                    const antGeo = new THREE.CylinderGeometry(0.024, 0.024, 0.36, 8);
                    const antMat = new THREE.MeshLambertMaterial({{ color: 0xaaaaaa }});
                    const ant = new THREE.Mesh(antGeo, antMat);
                    ant.position.set(0, 1.43, -0.3);
                    group.add(ant);

                    const lightGeo = new THREE.SphereGeometry(0.045, 8, 8);
                    const lightMat = new THREE.MeshBasicMaterial({{ color: 0xff0000 }});
                    const light = new THREE.Mesh(lightGeo, lightMat);
                    light.position.set(0, 1.61, -0.3);
                    group.add(light);
                }} else {{
                    const geo = new THREE.BoxGeometry(0.9, 0.9, 0.9);
                    const mat = new THREE.MeshLambertMaterial({{ color: colorHex }});
                    const mesh = new THREE.Mesh(geo, mat);
                    mesh.position.y = 0.45;
                    mesh.castShadow = true;
                    group.add(mesh);
                    group.userData.bodyMesh = mesh;
                }}

                group.position.set(agent.position[0], agent.position[1], agent.position[2]);
                scene.add(group);
                agentMeshes[agentKey] = group;

                // Add label
                const canvas = document.createElement('canvas');
                const context = canvas.getContext('2d');
                canvas.width = 256;
                canvas.height = 64;
                context.fillStyle = "#ffffff";
                context.fillRect(0, 0, canvas.width, canvas.height);
                context.fillStyle = "#000000";
                context.font = "24px Arial";
                context.textAlign = "center";
                context.fillText(agent.name, canvas.width/2, canvas.height/2 + 8);

                const texture = new THREE.CanvasTexture(canvas);
                const spriteMaterial = new THREE.SpriteMaterial({{ map: texture }});
                const sprite = new THREE.Sprite(spriteMaterial);
                sprite.position.set(agent.position[0], agent.position[1] + 1.5, agent.position[2]);
                sprite.scale.set(1.5, 0.4, 1);
                scene.add(sprite);
            }});

            // Add customer label
            const custCanvas = document.createElement('canvas');
            const custCtx = custCanvas.getContext('2d');
            custCanvas.width = 256;
            custCanvas.height = 64;
            custCtx.fillStyle = "#ffffff";
            custCtx.fillRect(0, 0, custCanvas.width, custCanvas.height);
            custCtx.fillStyle = "#000000";
            custCtx.font = "24px Arial";
            custCtx.textAlign = "center";
            custCtx.fillText("Customer", custCanvas.width/2, custCanvas.height/2 + 8);
            const custTex = new THREE.CanvasTexture(custCanvas);
            const custSpriteMat = new THREE.SpriteMaterial({{ map: custTex }});
            const custSprite = new THREE.Sprite(custSpriteMat);
            custSprite.position.set(-10, 2.3, 0);
            custSprite.scale.set(1.5, 0.4, 1);
            scene.add(custSprite);

            // Draw supply chain connection lines
            const chainNodes = [
                {{ name: 'customer', pos: [-10, 0.4, 0] }},
                {{ name: 'retailer', pos: [0, 0.4, 0] }},
                {{ name: 'wholesaler', pos: [10, 0.4, 0] }},
                {{ name: 'distributor', pos: [20, 0.4, 0] }},
                {{ name: 'manufacturer', pos: [30, 0.4, 0] }}
            ];
            for (let ci = 0; ci < chainNodes.length - 1; ci++) {{
                const from = new THREE.Vector3(...chainNodes[ci].pos);
                const to = new THREE.Vector3(...chainNodes[ci + 1].pos);
                const midPt = new THREE.Vector3().addVectors(from, to).multiplyScalar(0.5);

                // Ground connection line
                const lineGeo = new THREE.BufferGeometry().setFromPoints([from, to]);
                const lineMat = new THREE.LineBasicMaterial({{ color: 0x888888, transparent: true, opacity: 0.4 }});
                const line = new THREE.Line(lineGeo, lineMat);
                scene.add(line);
                line.userData = {{ from: chainNodes[ci].name, to: chainNodes[ci + 1].name }};
                agentMeshes['line_' + chainNodes[ci].name] = line;

                // Arrow indicator on ground
                const arrGeo = new THREE.ConeGeometry(0.18, 0.5, 4);
                const arrMat = new THREE.MeshLambertMaterial({{ color: 0x888888, transparent: true, opacity: 0.5 }});
                const arr = new THREE.Mesh(arrGeo, arrMat);
                arr.position.copy(midPt);
                arr.position.y = 0.08;
                arr.rotation.z = -Math.PI / 2;
                scene.add(arr);
                agentMeshes['arrow_' + chainNodes[ci].name] = arr;
            }}

            // Create floating info panels for each node
            createAllInfoPanels();
        }}

        // Info panel management
        let infoPanels = {{}};
        let infoPanelCanvases = {{}};

        function createAllInfoPanels() {{
            const nodePositions = {{
                'customer': [-10, -2.5, 0],
                'retailer': [0, -2.5, 0],
                'wholesaler': [10, -2.5, 0],
                'distributor': [20, -2.5, 0],
                'manufacturer': [30, -2.5, 0]
            }};
            const nodeColors = {{
                'customer': '#FECA57',
                'retailer': '#FF6B6B',
                'wholesaler': '#4ECDC4',
                'distributor': '#45B7D1',
                'manufacturer': '#96CEB4'
            }};

            Object.keys(nodePositions).forEach(key => {{
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = 280;
                canvas.height = 150;
                infoPanelCanvases[key] = {{ canvas: canvas, ctx: ctx }};

                const texture = new THREE.CanvasTexture(canvas);
                const material = new THREE.SpriteMaterial({{ map: texture }});
                const sprite = new THREE.Sprite(material);
                sprite.position.set(...nodePositions[key]);
                sprite.scale.set(4.5, 2.4, 1);
                scene.add(sprite);
                infoPanels[key] = sprite;
            }});
        }}

        function updateNodeInfoPanel(nodeKey, data) {{
            if (!infoPanelCanvases[nodeKey]) return;
            const {{ canvas, ctx }} = infoPanelCanvases[nodeKey];
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            // Background
            ctx.fillStyle = 'rgba(0,0,0,0.75)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.strokeStyle = 'rgba(255,255,255,0.5)';
            ctx.lineWidth = 1;
            ctx.strokeRect(2, 2, canvas.width-4, canvas.height-4);

            // Title
            ctx.fillStyle = 'white';
            ctx.font = 'bold 13px Arial';
            ctx.textAlign = 'center';
            const titles = {{ customer: 'Customer', retailer: 'Retailer', wholesaler: 'Wholesaler', distributor: 'Distributor', manufacturer: 'Manufacturer' }};
            ctx.fillText(titles[nodeKey] || nodeKey, canvas.width/2, 18);

            // Data
            ctx.font = '11px Arial';
            ctx.textAlign = 'left';
            const inv = data.inventory || 0;
            const bo = data.backorder || 0;
            const ord = data.order_placed || 0;
            const transit = data.in_transit || 0;
            const demand = data.demand_received || 0;

            ctx.fillStyle = inv > 0 ? '#4CAF50' : (bo > 0 ? '#F44336' : '#FFC107');
            ctx.fillText('Net Inv: ' + (inv - bo), 8, 38);

            ctx.fillStyle = '#E0E0E0';
            ctx.fillText('Stock: ' + inv + ' | Back: ' + bo, 8, 55);
            ctx.fillText('Order: ' + ord + ' | Demand: ' + demand, 8, 72);
            ctx.fillText('In-Transit: ' + transit, 8, 89);

            // Cost info
            const holdingCost = (inv * 0.5).toFixed(1);
            const shortageCost = (bo * 1.0).toFixed(1);
            ctx.fillStyle = '#4CAF50';
            ctx.fillText('Holding: ' + holdingCost, 8, 110);
            ctx.fillStyle = '#F44336';
            ctx.fillText('Shortage: ' + shortageCost, 150, 110);

            // Shipment pipeline
            if (data.shipment_pipeline && data.shipment_pipeline.length > 0) {{
                ctx.fillStyle = '#FFD700';
                const pipeline = data.shipment_pipeline.join(', ');
                ctx.fillText('Pipeline: [' + pipeline + ']', 8, 130);
            }}

            infoPanels[nodeKey].material.map.needsUpdate = true;
        }}
        
        // Initialize controls
        function initControls() {{
            const roundSlider = document.getElementById('roundSlider');
            const playButton = document.getElementById('playButton');
            const speedSlider = document.getElementById('speedSlider');
            const resetButton = document.getElementById('resetButton');
            
            // Set round slider maximum
            roundSlider.max = simulationData.rounds.length - 1;
            
            // Round slider event
            roundSlider.addEventListener('input', (e) => {{
                currentRoundIndex = parseInt(e.target.value);
                updateVisualization(currentRoundIndex);
                updateRoundDisplay();
            }});
            
            // Play button event
            playButton.addEventListener('click', () => {{
                if (isPlaying) {{
                    stopAnimation();
                }} else {{
                    startAnimation();
                }}
            }});
            
            // Speed slider event
            speedSlider.addEventListener('input', (e) => {{
                playSpeed = parseInt(e.target.value);
                document.getElementById('speedDisplay').textContent = playSpeed + 'x';
            }});
            
            // Reset button event
            resetButton.addEventListener('click', () => {{
                camera.position.set(5, 8, 12);
                controls.reset();
            }});
        }}
        
        // Update visualization
        function updateVisualization(roundIndex) {{
            if (roundIndex >= simulationData.rounds.length) return;
            
            const roundData = simulationData.rounds[roundIndex];
            
            // Update info panel
            updateInfoPanel(roundData);

            // Clear previous order flow
            orderFlows.forEach(flow => scene.remove(flow));
            orderFlows = [];

            // Update demand display in info panel only
            const demand = roundData.customer_demand || roundData.demand || 0;
            console.log("Demand updated to: " + demand);
            
            // Update agent status (via scale and color changes)
            Object.keys(roundData.agents).forEach(agentKey => {{
                const agentData = roundData.agents[agentKey];
                const group = agentMeshes[agentKey];

                if (group) {{
                    // Adjust size based on inventory
                    const inventoryScale = Math.max(0.5, Math.min(2.0, 1 + agentData.inventory * 0.1));
                    group.scale.set(inventoryScale, inventoryScale, inventoryScale);

                    // Adjust color based on net inventory status
                    const netInv = (agentData.inventory || 0) - (agentData.backorder || 0);
                    let statusColor;
                    if (netInv > 0) {{
                        statusColor = 0x2196F3;  // Blue - healthy
                    }} else if (netInv === 0) {{
                        statusColor = 0xFFC107;  // Yellow - warning
                    }} else {{
                        statusColor = 0xF44336;  // Red - backorder
                    }}

                    if (group.userData.bodyMesh) {{
                        group.userData.bodyMesh.material.color.setHex(statusColor);
                        group.userData.bodyMesh.material.needsUpdate = true;
                    }}
                }}

                // Update info panel
                updateNodeInfoPanel(agentKey, agentData);
            }});

            // Update customer info panel
            const custDemand = roundData.customer_demand || roundData.demand || 0;
            updateNodeInfoPanel('customer', {{
                inventory: 0, backorder: 0, order_placed: 0,
                in_transit: 0, demand_received: custDemand
            }});

            // Create order flow animation (upstream: retailer→wholesaler→distributor→manufacturer)
            roundData.orders_flow.forEach(order => {{
                createOrderFlow(order.from, order.to, order.quantity);
            }});

            // Create shipment flow animation (downstream: manufacturer→distributor→wholesaler→retailer)
            Object.keys(roundData.agents).forEach(agentKey => {{
                const agentData = roundData.agents[agentKey];
                if (agentData.incoming_shipment && agentData.incoming_shipment > 0) {{
                    const downstreamMap = {{
                        'manufacturer': 'distributor',
                        'distributor': 'wholesaler',
                        'wholesaler': 'retailer'
                    }};
                    if (downstreamMap[agentKey]) {{
                        createShipmentFlow(agentKey, downstreamMap[agentKey], agentData.incoming_shipment);
                    }}
                }}
            }});

            // Pulse connection lines based on activity
            Object.keys(roundData.agents).forEach(agentKey => {{
                const lineKey = 'line_' + (agentKey === 'manufacturer' ? 'distributor' :
                    agentKey === 'distributor' ? 'wholesaler' :
                    agentKey === 'wholesaler' ? 'retailer' :
                    agentKey === 'retailer' ? 'customer' : '');
                const line = agentMeshes[lineKey];
                if (line) {{
                    const agentData = roundData.agents[agentKey];
                    const hasActivity = (agentData.order_placed > 0) || (agentData.incoming_shipment > 0);
                    line.material.opacity = hasActivity ? 0.9 : 0.3;
                    line.material.color.setHex(hasActivity ? 0x00AA00 : 0x888888);
                }}
            }});

            // Update info panel
            updateInfoPanel(roundData);
        }}
        
        // Create order flow
        function createOrderFlow(fromAgent, toAgent, quantity) {{
            const fromPos = getAgentPosition(fromAgent);
            const toPos = getAgentPosition(toAgent);
            
            if (!fromPos || !toPos) return;
            
            // Use different colors based on sender role
            let color = 0x00ff00; // Default green
            const roleColors = {{
                'customer': 0xFECA57,
                'retailer': 0xFF6B6B,
                'wholesaler': 0x4ECDC4,
                'distributor': 0x45B7D1,
                'manufacturer': 0x96CEB4
            }};
            if (roleColors[fromAgent]) {{
                color = roleColors[fromAgent];
            }}
            
            // Create arrow geometry
            const direction = new THREE.Vector3().subVectors(toPos, fromPos).normalize();
            const arrowGeometry = new THREE.ConeGeometry(0.15, 0.8, 8);
            const arrowMaterial = new THREE.MeshLambertMaterial({{ color: color }});
            const arrow = new THREE.Mesh(arrowGeometry, arrowMaterial);
            
            // Set arrow position and direction
            const midPoint = new THREE.Vector3().addVectors(fromPos, toPos).multiplyScalar(0.5);
            arrow.position.copy(midPoint);
            arrow.position.y += 2;
            arrow.lookAt(toPos);
            arrow.rotateX(Math.PI / 2);
            
            // Add quantity label
            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.width = 128;
            canvas.height = 64;
            context.fillStyle = "#ffffff";
            context.fillRect(0, 0, canvas.width, canvas.height);
            context.fillStyle = "#000000";
            context.font = "20px Arial";
            context.textAlign = "center";
            context.fillText(quantity.toString(), canvas.width/2, canvas.height/2 + 8);
            
            const texture = new THREE.CanvasTexture(canvas);
            const spriteMaterial = new THREE.SpriteMaterial({{ map: texture }});
            const sprite = new THREE.Sprite(spriteMaterial);
            sprite.position.copy(midPoint);
            sprite.position.y += 2.5;
            sprite.scale.set(1, 0.5, 1);
            
            // Add animation effect
            let progress = 0;
            const animateArrow = () => {{
                progress += 0.05;
                if (progress <= 1) {{
                    const currentPos = new THREE.Vector3().lerpVectors(fromPos, toPos, progress);
                    currentPos.y += Math.sin(progress * Math.PI) * 1; // Arc trajectory
                    arrow.position.copy(currentPos);
                    arrow.position.y += 2;
                    sprite.position.copy(currentPos);
                    sprite.position.y += 2.5;
                    requestAnimationFrame(animateArrow);
                }} else {{
                    scene.remove(arrow);
                    scene.remove(sprite);
                    const index = orderFlows.indexOf(arrow);
                    if (index > -1) {{
                        orderFlows.splice(index, 2);
                    }}
                }}
            }};
            
            scene.add(arrow);
            scene.add(sprite);
            orderFlows.push(arrow, sprite);
            animateArrow();
        }}

        // Create shipment flow (downstream goods movement)
        function createShipmentFlow(fromAgent, toAgent, quantity) {{
            const fromPos = getAgentPosition(fromAgent);
            const toPos = getAgentPosition(toAgent);

            if (!fromPos || !toPos || quantity <= 0) return;

            const pkgGeo = new THREE.BoxGeometry(0.35, 0.35, 0.35);
            const pkgMat = new THREE.MeshLambertMaterial({{ color: 0x8B4513 }});
            const pkg = new THREE.Mesh(pkgGeo, pkgMat);
            pkg.position.copy(fromPos);
            pkg.position.y += 1.5;
            pkg.castShadow = true;
            scene.add(pkg);

            const canvas = document.createElement('canvas');
            const context = canvas.getContext('2d');
            canvas.width = 128;
            canvas.height = 40;
            context.fillStyle = 'rgba(139,69,19,0.85)';
            context.fillRect(0, 0, canvas.width, canvas.height);
            context.fillStyle = 'white';
            context.font = 'bold 16px Arial';
            context.textAlign = 'center';
            context.fillText('Ship: ' + quantity, canvas.width/2, canvas.height/2 + 6);

            const texture = new THREE.CanvasTexture(canvas);
            const spriteMaterial = new THREE.SpriteMaterial({{ map: texture }});
            const label = new THREE.Sprite(spriteMaterial);
            label.position.copy(fromPos);
            label.position.y += 2.2;
            label.scale.set(2, 0.6, 1);
            scene.add(label);

            const startTime = Date.now();
            const duration = 2500;

            const animateShipment = () => {{
                const elapsed = Date.now() - startTime;
                const progress = Math.min(elapsed / duration, 1);
                const easeProgress = 1 - Math.pow(1 - progress, 2);

                const currentPos = new THREE.Vector3().lerpVectors(fromPos, toPos, easeProgress);
                currentPos.y += 1.5 + Math.sin(progress * Math.PI) * 0.8;
                pkg.position.copy(currentPos);
                label.position.copy(currentPos);
                label.position.y += 0.7;
                pkg.rotation.x += 0.08;
                pkg.rotation.z += 0.04;

                if (progress < 1) {{
                    requestAnimationFrame(animateShipment);
                }} else {{
                    scene.remove(pkg);
                    scene.remove(label);
                }}
            }};

            orderFlows.push(pkg, label);
            animateShipment();
        }}

        // Get agent position
        function getAgentPosition(agentKey) {{
            if (agentKey === 'customer') {{
                return new THREE.Vector3(-10, 0.75, 0);
            }}
            if (simulationData.agents[agentKey]) {{
                const pos = simulationData.agents[agentKey].position;
                return new THREE.Vector3(pos[0], pos[1] + 0.75, pos[2]);
            }}
            return null;
        }}
        
        // Update info panel
        function updateInfoPanel(roundData) {{
            document.getElementById('currentRound').textContent = roundData.round;
            // Use customer_demand field to display market demand
            document.getElementById('currentDemand').textContent = roundData.customer_demand || roundData.demand || 0;
            
            let totalInventory = 0;
            let totalInTransit = 0;
            let totalBackorder = 0;
            
            // Update participant data
            if (roundData.agents.retailer) {{
                const retailer = roundData.agents.retailer;
                document.getElementById('retailerDemand').textContent = retailer.demand_received || 0;
                document.getElementById('retailerInventory').textContent = retailer.inventory || 0;
                document.getElementById('retailerBackorder').textContent = retailer.backorder || 0;
                document.getElementById('retailerOrder').textContent = retailer.order_placed || 0;
            }}
            
            if (roundData.agents.wholesaler) {{
                const wholesaler = roundData.agents.wholesaler;
                document.getElementById('wholesalerDemand').textContent = wholesaler.demand_received || 0;
                document.getElementById('wholesalerInventory').textContent = wholesaler.inventory || 0;
                document.getElementById('wholesalerBackorder').textContent = wholesaler.backorder || 0;
                document.getElementById('wholesalerOrder').textContent = wholesaler.order_placed || 0;
            }}

            if (roundData.agents.distributor) {{
                const distributor = roundData.agents.distributor;
                document.getElementById('distributorDemand').textContent = distributor.demand_received || 0;
                document.getElementById('distributorInventory').textContent = distributor.inventory || 0;
                document.getElementById('distributorBackorder').textContent = distributor.backorder || 0;
                document.getElementById('distributorOrder').textContent = distributor.order_placed || 0;
            }}

            if (roundData.agents.manufacturer) {{
                const manufacturer = roundData.agents.manufacturer;
                document.getElementById('manufacturerDemand').textContent = manufacturer.demand_received || 0;
                document.getElementById('manufacturerInventory').textContent = manufacturer.inventory || 0;
                document.getElementById('manufacturerBackorder').textContent = manufacturer.backorder || 0;
                document.getElementById('manufacturerOrder').textContent = manufacturer.order_placed || 0;
            }}
            
            // Calculate totals
            Object.values(roundData.agents).forEach(agent => {{
                totalInventory += agent.inventory || 0;
                totalInTransit += agent.in_transit || 0;
                totalBackorder += agent.backorder || 0;
            }});
            
            document.getElementById('totalInventory').textContent = totalInventory;
            document.getElementById('totalInTransit').textContent = totalInTransit;
            document.getElementById('totalBackorder').textContent = totalBackorder;
        }}
        
        // Start animation
        function startAnimation() {{
            isPlaying = true;
            document.getElementById('playButton').textContent = '⏸️ Pause';
            playAnimation();
        }}
        
        // Stop animation
        function stopAnimation() {{
            isPlaying = false;
            document.getElementById('playButton').textContent = '▶️ Play';
            if (animationId) {{
                clearTimeout(animationId);
            }}
        }}
        
        // Play animation
        function playAnimation() {{
            if (!isPlaying) return;
            
            if (currentRoundIndex < simulationData.rounds.length - 1) {{
                currentRoundIndex++;
                updateVisualization(currentRoundIndex);
                updateRoundDisplay();
                document.getElementById('roundSlider').value = currentRoundIndex;
                
                animationId = setTimeout(playAnimation, 1000 / playSpeed);
            }} else {{
                stopAnimation();
            }}
        }}
        
        // Update round display
        function updateRoundDisplay() {{
            document.getElementById('roundDisplay').textContent = 'Round ' + currentRoundIndex;
        }}
        
        // Render loop
        function animate() {{
            requestAnimationFrame(animate);
            controls.update();
            
            // Animate customer to look toward supply chain
            const customerGroup = agentMeshes['customer'];
            if (customerGroup && customerGroup.children[0]) {{
                // Subtle idle animation - slight head turn
                const head = customerGroup.children[0];
                head.rotation.y = Math.sin(Date.now() / 2000) * 0.2;
            }}
            
            renderer.render(scene, camera);
        }}
        
        // Window resize
        window.addEventListener('resize', () => {{
            camera.aspect = {width} / {height};
            camera.updateProjectionMatrix();
            renderer.setSize({width}, {height});
        }});
        
        // Initialize scene
        // Defer initialization until Three.js + OrbitControls are loaded
        window._bootScene = function() {{ initScene(); }};

    </script>
</body>
</html>
        """
        
        return html_template
    
    def render_in_streamlit(self, simulation_result, width: int = 1200, height: int = 800):
        """Render 3D visualization in Streamlit"""
        html_content = self.create_3d_visualization(simulation_result, width, height)
        components.html(html_content, width=width, height=height)
    
    def save_html(self, simulation_result, output_path: str, width: int = 1200, height: int = 800):
        """Save HTML file"""
        html_content = self.create_3d_visualization(simulation_result, width, height)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"3D visualization saved to: {output_path}")