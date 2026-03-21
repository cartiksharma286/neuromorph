#!/usr/bin/env python3
"""
Flask App Integration for Quantum Noise Reduction
==================================================

Adds quantum-based noise reduction endpoints to the Flask MRI simulator.
Provides real-time reconstruction with multiple filter options.
"""

from flask import Flask, jsonify, request
import numpy as np
from quantum_noise_reduction import (
    QuantumSignalReconstructor,
    WienerFilter,
    QuantumMachineLearningArtifactDetector
)
from quantum_reconstruction_pipeline import ComparativeNoiseReductionAnalysis
import base64
import io
from PIL import Image


def create_quantum_noise_reduction_blueprints(app: Flask):
    """
    Register quantum noise reduction endpoints with Flask app.
    
    Endpoints:
      POST /quantum/reconstruct     - Full quantum reconstruction
      POST /quantum/wiener          - Wiener filter only
      POST /quantum/qml             - Quantum ML only
      POST /quantum/compare         - Compare all methods
    """
    
    @app.route('/quantum/reconstruct', methods=['POST'])
    def quantum_full_reconstruction():
        """
        Perform full quantum signal reconstruction.
        
        Request JSON:
          {
            'image_base64': <base64 encoded image>,
            'method': 'full' | 'wiener' | 'qml' | 'speckle'
          }
        """
        try:
            data = request.json
            
            # Decode image from base64
            image_data = base64.b64decode(data.get('image_base64', ''))
            image = np.array(Image.open(io.BytesIO(image_data)).convert('L'), dtype=float)
            image = image / 255.0  # Normalize to [0, 1]
            
            method = data.get('method', 'full')
            
            # Perform reconstruction
            reconstructor = QuantumSignalReconstructor()
            results = reconstructor.reconstruct(image, method=method)
            
            # Encode results
            reconstructed = results['final_reconstructed']
            reconstructed_normalized = (reconstructed / np.max(reconstructed) * 255).astype(np.uint8)
            
            img_resized = Image.fromarray(reconstructed_normalized)
            img_buffer = io.BytesIO()
            img_resized.save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            return jsonify({
                'success': True,
                'reconstructed_image': img_base64,
                'metrics': {
                    'SNR_dB': float(results['metrics']['SNR_dB']),
                    'PSNR_dB': float(results['metrics']['PSNR_dB']),
                    'Noise_Reduction_Factor': float(results['metrics']['Noise_Reduction_Factor']),
                    'MSE': float(results['metrics']['MSE'])
                },
                'method_used': method
            })
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 400
    
    @app.route('/quantum/wiener', methods=['POST'])
    def wiener_filter_endpoint():
        """Apply Wiener filter to image."""
        try:
            data = request.json
            
            # Decode image
            image_data = base64.b64decode(data.get('image_base64', ''))
            image = np.array(Image.open(io.BytesIO(image_data)).convert('L'), dtype=float)
            image = image / 255.0
            
            # Apply Wiener filter
            wiener = WienerFilter()
            filtered = wiener.filter_2d(image)
            
            # Encode result
            filtered_normalized = (filtered / np.max(filtered) * 255).astype(np.uint8)
            img_buffer = io.BytesIO()
            Image.fromarray(filtered_normalized).save(img_buffer, format='PNG')
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode()
            
            return jsonify({
                'success': True,
                'filtered_image': img_base64,
                'method': 'Wiener Filter (MMSE Optimal)'
            })
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 400
    
    @app.route('/quantum/qml', methods=['POST'])
    def quantum_ml_endpoint():
        """Apply Quantum ML artifact detection and removal."""
        try:
            data = request.json
            
            # Decode image
            image_data = base64.b64decode(data.get('image_base64', ''))
            image = np.array(Image.open(io.BytesIO(image_data)).convert('L'), dtype=float)
            image = image / 255.0
            
            # Apply QML
            qml = QuantumMachineLearningArtifactDetector(n_clusters=4)
            artifact_mask, cluster_map = qml.cluster_artifacts(image)
            reconstructed = qml.quantum_reconstruction(image, artifact_mask)
            
            # Encode results
            recon_normalized = (reconstructed / np.max(reconstructed) * 255).astype(np.uint8)
            artifact_normalized = (artifact_mask.astype(float) * 255).astype(np.uint8)
            
            # Create images
            recon_buffer = io.BytesIO()
            Image.fromarray(recon_normalized).save(recon_buffer, format='PNG')
            recon_base64 = base64.b64encode(recon_buffer.getvalue()).decode()
            
            artifact_buffer = io.BytesIO()
            Image.fromarray(artifact_normalized).save(artifact_buffer, format='PNG')
            artifact_base64 = base64.b64encode(artifact_buffer.getvalue()).decode()
            
            return jsonify({
                'success': True,
                'reconstructed_image': recon_base64,
                'artifact_mask': artifact_base64,
                'artifacts_detected': int(np.sum(artifact_mask)),
                'artifact_percentage': float(np.sum(artifact_mask) / artifact_mask.size * 100),
                'method': 'Quantum Machine Learning'
            })
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 400
    
    @app.route('/quantum/compare', methods=['POST'])
    def compare_methods_endpoint():
        """Compare all noise reduction methods."""
        try:
            data = request.json
            
            # Decode image
            image_data = base64.b64decode(data.get('image_base64', ''))
            image = np.array(Image.open(io.BytesIO(image_data)).convert('L'), dtype=float)
            image = image / 255.0
            
            # Compare methods
            analyzer = ComparativeNoiseReductionAnalysis()
            comparison = analyzer.compare_methods(image)
            
            # Encode all results
            results_dict = {}
            for method in ['wiener', 'qml', 'speckle', 'combined']:
                result_img = comparison[method]['result']
                normalized = (result_img / np.max(result_img) * 255).astype(np.uint8)
                
                buffer = io.BytesIO()
                Image.fromarray(normalized).save(buffer, format='PNG')
                base64_img = base64.b64encode(buffer.getvalue()).decode()
                
                results_dict[method] = {
                    'image': base64_img,
                    'SNR_dB': float(comparison[method]['snr']),
                    'PSNR_dB': float(comparison[method]['psnr'])
                }
            
            return jsonify({
                'success': True,
                'methods': results_dict,
                'best_method': max(
                    results_dict.keys(),
                    key=lambda k: results_dict[k]['SNR_dB']
                )
            })
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 400
    
    @app.route('/quantum/info', methods=['GET'])
    def quantum_info():
        """Get information about quantum noise reduction methods."""
        return jsonify({
            'name': 'Quantum Signal Reconstruction Pipeline',
            'version': '1.0',
            'methods': {
                'wiener': {
                    'description': 'Wiener filter with MMSE-optimal estimation',
                    'advantages': 'Optimal for Gaussian noise, computationally efficient',
                    'best_for': 'Gaussian thermal noise from receiver coils'
                },
                'qml': {
                    'description': 'Unsupervised quantum machine learning artifact detection',
                    'advantages': 'Detects complex artifact signatures without labels',
                    'best_for': 'Multiplicative speckle noise and structured artifacts'
                },
                'speckle': {
                    'description': 'Adaptive Lee and homomorphic filtering for speckle',
                    'advantages': 'Preserves edges and structure',
                    'best_for': 'MRI speckle noise (Rician/Rice-Rayleigh)'
                },
                'full': {
                    'description': 'Combined pipeline using all three methods sequentially',
                    'advantages': 'Best overall SNR improvement, comprehensive artifact removal',
                    'best_for': 'Complex real-world MRI images with mixed noise'
                }
            }
        })


def create_quantum_analysis_dashboard():
    """
    Create HTML dashboard for quantum noise reduction analysis.
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Quantum Signal Reconstruction</title>
        <style>
            body { font-family: Arial; margin: 20px; background: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
            h1 { color: #1f4788; border-bottom: 3px solid #1f4788; padding-bottom: 10px; }
            .method-box { 
                border: 1px solid #ddd; 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 5px;
                background: #f9f9f9;
            }
            .method-title { font-weight: bold; color: #1f4788; font-size: 1.1em; }
            .metric { margin: 8px 0; }
            .metric-value { color: #e74c3c; font-weight: bold; }
            .upload-section { margin: 20px 0; }
            button { 
                background: #1f4788; 
                color: white; 
                padding: 10px 20px; 
                border: none; 
                border-radius: 4px; 
                cursor: pointer;
                font-size: 1em;
            }
            button:hover { background: #16365d; }
            .results { margin-top: 20px; }
            .image-grid {
                display: grid;
                grid-template-columns: repeat(2, 1fr);
                gap: 15px;
                margin: 20px 0;
            }
            .image-box { 
                text-align: center; 
                padding: 10px; 
                background: #f0f0f0; 
                border-radius: 5px;
            }
            .image-box img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>⚛️ Quantum Signal Reconstruction Pipeline</h1>
            
            <div class="method-box">
                <div class="method-title">Pipeline Overview</div>
                <p>Advanced multi-stage noise reduction combining:</p>
                <ul>
                    <li><strong>Wiener Filter:</strong> MMSE-optimal linear filtering for Gaussian noise</li>
                    <li><strong>Quantum ML:</strong> Unsupervised artifact detection via quantum-inspired clustering</li>
                    <li><strong>Speckle Filter:</strong> Lee filter adaptive suppression for multiplicative noise</li>
                </ul>
            </div>
            
            <div class="upload-section">
                <h2>Upload Image for Analysis</h2>
                <input type="file" id="imageInput" accept="image/*">
                <button onclick="performQuantumReconstruction()">Run Full Reconstruction</button>
                <button onclick="compareAllMethods()">Compare All Methods</button>
            </div>
            
            <div class="results" id="results"></div>
        </div>
        
        <script>
            async function performQuantumReconstruction() {
                const file = document.getElementById('imageInput').files[0];
                if (!file) {
                    alert('Please select an image');
                    return;
                }
                
                const reader = new FileReader();
                reader.onload = async (e) => {
                    const base64 = btoa(new Uint8Array(e.target.result)
                        .reduce((data, byte) => data + String.fromCharCode(byte), ''));
                    
                    const response = await fetch('/quantum/reconstruct', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            image_base64: base64,
                            method: 'full'
                        })
                    });
                    
                    const result = await response.json();
                    if (result.success) {
                        displayResults(result);
                    }
                };
                reader.readAsArrayBuffer(file);
            }
            
            async function compareAllMethods() {
                const file = document.getElementById('imageInput').files[0];
                if (!file) {
                    alert('Please select an image');
                    return;
                }
                
                const reader = new FileReader();
                reader.onload = async (e) => {
                    const base64 = btoa(new Uint8Array(e.target.result)
                        .reduce((data, byte) => data + String.fromCharCode(byte), ''));
                    
                    const response = await fetch('/quantum/compare', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            image_base64: base64
                        })
                    });
                    
                    const result = await response.json();
                    if (result.success) {
                        displayComparison(result);
                    }
                };
                reader.readAsArrayBuffer(file);
            }
            
            function displayResults(data) {
                const resultsDiv = document.getElementById('results');
                let html = '<h2>Reconstruction Results</h2>';
                html += '<div class="image-grid">';
                html += `<div class="image-box"><img src="data:image/png;base64,${data.reconstructed_image}"><p>Reconstructed</p></div>`;
                html += '</div>';
                html += '<div class="method-box">';
                html += '<div class="method-title">Quality Metrics</div>';
                for (const [key, value] of Object.entries(data.metrics)) {
                    html += `<div class="metric">${key}: <span class="metric-value">${value.toFixed(3)}</span></div>`;
                }
                html += '</div>';
                resultsDiv.innerHTML = html;
            }
            
            function displayComparison(data) {
                const resultsDiv = document.getElementById('results');
                let html = '<h2>Method Comparison</h2>';
                html += '<div class="image-grid">';
                
                for (const [method, info] of Object.entries(data.methods)) {
                    html += `
                        <div class="image-box">
                            <img src="data:image/png;base64,${info.image}">
                            <p><strong>${method.toUpperCase()}</strong></p>
                            <p>SNR: ${info.SNR_dB.toFixed(2)} dB</p>
                        </div>
                    `;
                }
                
                html += '</div>';
                html += `<div class="method-box"><strong>Best Method:</strong> ${data.best_method.toUpperCase()}</div>`;
                resultsDiv.innerHTML = html;
            }
        </script>
    </body>
    </html>
    """
    return html


if __name__ == '__main__':
    print("✓ Quantum noise reduction Flask integration ready")
    print("✓ Add to app.py: import quantum_app_integration")
    print("✓ Then call: quantum_app_integration.create_quantum_noise_reduction_blueprints(app)")
