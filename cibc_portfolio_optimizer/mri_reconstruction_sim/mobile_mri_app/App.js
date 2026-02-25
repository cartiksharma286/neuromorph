
import React, { useState } from 'react';
import { StyleSheet, Text, View, Image, TouchableOpacity, ScrollView, Alert, Platform } from 'react-native';
import { StatusBar } from 'expo-status-bar';

// Define the backend URL based on platform
// For Android Emulator, use 10.0.2.2. For iOS Sim, use localhost. For physical device, use LAN IP.
const SERVER_URL = Platform.OS === 'android' ? 'http://10.0.2.2:5050' : 'http://localhost:5050';

export default function App() {
    const [imgRecon, setImgRecon] = useState(null);
    const [metrics, setMetrics] = useState(null);
    const [loading, setLoading] = useState(false);
    const [sequence, setSequence] = useState('QuantumGenerativeRecon');

    const runSimulation = async () => {
        setLoading(true);
        try {
            const response = await fetch(`${SERVER_URL}/api/simulate`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    sequence: sequence,
                    coils: 'quantum_vascular',
                    recon_method: 'QuantumML',
                    resolution: 128,
                    noise: 0.05
                }),
            });

            const data = await response.json();
            if (data.success) {
                setImgRecon(data.plots.recon); // Base64 string
                setMetrics(data.metrics);
            } else {
                Alert.alert('Error', 'Simulation failed on server.');
            }
        } catch (error) {
            console.error(error);
            Alert.alert('Network Error', `Could not connect to ${SERVER_URL}. Ensure Flask is running.`);
        } finally {
            setLoading(false);
        }
    };

    const sequences = [
        { label: 'Quantum Generative (QML)', value: 'QuantumGenerativeRecon' },
        { label: 'Statistical Bayesian', value: 'StatisticalBayesianInference' },
        { label: 'Quantum NVQLink', value: 'QuantumNVQLink' },
        { label: 'Gemini 3.0 Context', value: 'Gemini3.0' },
    ];

    return (
        <View style={styles.container}>
            <StatusBar style="light" />

            {/* Header */}
            <View style={styles.header}>
                <Text style={styles.headerTitle}>NeuroPulse Mobile</Text>
                <Text style={styles.headerSubtitle}>v1.0 Quantum Edition</Text>
            </View>

            <ScrollView contentContainerStyle={styles.scrollContent}>

                {/* Sequence Selector */}
                <Text style={styles.sectionTitle}>Select Pulse Sequence</Text>
                <View style={styles.selectorContainer}>
                    {sequences.map((seq) => (
                        <TouchableOpacity
                            key={seq.value}
                            style={[styles.seqButton, sequence === seq.value && styles.seqButtonActive]}
                            onPress={() => setSequence(seq.value)}
                        >
                            <Text style={[styles.seqText, sequence === seq.value && styles.seqTextActive]}>
                                {seq.label}
                            </Text>
                        </TouchableOpacity>
                    ))}
                </View>

                {/* Action Button */}
                <TouchableOpacity
                    style={styles.actionButton}
                    onPress={runSimulation}
                    disabled={loading}
                >
                    <Text style={styles.actionButtonText}>
                        {loading ? 'Simulating...' : 'Run Quantum Reconstruction'}
                    </Text>
                </TouchableOpacity>

                {/* Results Area */}
                {imgRecon && (
                    <View style={styles.resultContainer}>
                        <Text style={styles.resultTitle}>Reconstructed Image</Text>
                        <Image
                            source={{ uri: `data:image/png;base64,${imgRecon}` }}
                            style={styles.reconImage}
                            resizeMode="contain"
                        />

                        {/* Metrics */}
                        {metrics && (
                            <View style={styles.metricsGrid}>
                                <View style={styles.metricItem}>
                                    <Text style={styles.metricVal}>{metrics.snr_estimate?.toFixed(1)}</Text>
                                    <Text style={styles.metricLabel}>SNR</Text>
                                </View>
                                <View style={styles.metricItem}>
                                    <Text style={styles.metricVal}>{metrics.sharpness?.toFixed(1)}</Text>
                                    <Text style={styles.metricLabel}>Sharpness</Text>
                                </View>
                                <View style={styles.metricItem}>
                                    <Text style={styles.metricVal}>{metrics.contrast?.toFixed(2)}</Text>
                                    <Text style={styles.metricLabel}>Contrast</Text>
                                </View>
                                <View style={styles.metricItem}>
                                    <Text style={styles.metricVal}>{(metrics.entropy || 0).toFixed(2)}</Text>
                                    <Text style={styles.metricLabel}>Entropy</Text>
                                </View>
                            </View>
                        )}
                    </View>
                )}

            </ScrollView>
        </View>
    );
}

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: '#020617',
    },
    header: {
        paddingTop: 60,
        paddingBottom: 20,
        paddingHorizontal: 20,
        backgroundColor: '#0f172a',
        borderBottomWidth: 1,
        borderBottomColor: '#1e293b',
    },
    headerTitle: {
        fontSize: 24,
        fontWeight: 'bold',
        color: '#38bdf8',
    },
    headerSubtitle: {
        fontSize: 12,
        color: '#94a3b8',
        marginTop: 4,
        textTransform: 'uppercase',
        letterSpacing: 1,
    },
    scrollContent: {
        padding: 20,
    },
    sectionTitle: {
        fontSize: 16,
        color: '#e2e8f0',
        marginBottom: 12,
        fontWeight: '600',
    },
    selectorContainer: {
        marginBottom: 24,
    },
    seqButton: {
        paddingVertical: 12,
        paddingHorizontal: 16,
        backgroundColor: '#1e293b',
        borderRadius: 8,
        marginBottom: 8,
        borderWidth: 1,
        borderColor: '#334155',
    },
    seqButtonActive: {
        backgroundColor: '#0f172a',
        borderColor: '#38bdf8',
    },
    seqText: {
        color: '#94a3b8',
        fontSize: 14,
    },
    seqTextActive: {
        color: '#38bdf8',
        fontWeight: '600',
    },
    actionButton: {
        backgroundColor: '#38bdf8',
        padding: 16,
        borderRadius: 12,
        alignItems: 'center',
        marginBottom: 32,
        shadowColor: '#38bdf8',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.3,
        shadowRadius: 8,
        elevation: 8,
    },
    actionButtonText: {
        color: '#020617',
        fontSize: 16,
        fontWeight: 'bold',
        textTransform: 'uppercase',
    },
    resultContainer: {
        backgroundColor: '#0f172a',
        borderRadius: 16,
        padding: 16,
        borderWidth: 1,
        borderColor: '#1e293b',
    },
    resultTitle: {
        color: '#e2e8f0',
        fontSize: 14,
        marginBottom: 12,
        textAlign: 'center',
    },
    reconImage: {
        width: '100%',
        aspectRatio: 1,
        backgroundColor: '#000',
        borderRadius: 8,
        marginBottom: 16,
    },
    metricsGrid: {
        flexDirection: 'row',
        flexWrap: 'wrap',
        justifyContent: 'space-between',
        gap: 8,
    },
    metricItem: {
        width: '48%',
        backgroundColor: '#1e293b',
        padding: 12,
        borderRadius: 8,
        alignItems: 'center',
        marginBottom: 8,
    },
    metricVal: {
        color: '#38bdf8',
        fontSize: 18,
        fontWeight: '700',
    },
    metricLabel: {
        color: '#64748b',
        fontSize: 10,
        textTransform: 'uppercase',
        marginTop: 4,
    },
});
