
# NeuroPulse Mobile - React Native Client

This directory contains the cross-platform (iOS/Android) mobile application for the MRI Reconstruction Simulator.

## Architecture
This app is a client-side interface built with **React Native (Expo)**. It communicates with the Python Flask backend (`app.py`) running on your machine to perform the heavy MRI physics simulations.

## Prerequisites
- **Node.js**: Installed on your system.
- **Expo Go App**: Installed on your physical iOS or Android device.
- **Flask Backend running**: Ensure `mri_reconstruction_sim/app.py` is running on port 5050.

## Setup Instructions

1. **Install Dependencies**:
   Open a terminal in this directory (`mobile_mri_app`) and run:
   ```bash
   npm install
   ```

2. **Configure Network**:
   - The app tries to connect to `http://localhost:5050` (iOS Simulator) or `http://10.0.2.2:5050` (Android Emulator).
   - **Scanning QR Code (Physical Device)**: If running on a physical phone, you must verify your computer and phone are on the same Wi-Fi. 
     - Open `App.js` 
     - Change `const SERVER_URL` to your computer's local IP address (e.g., `http://192.168.1.15:5050`).

3. **Run the App**:
   ```bash
   npx expo start
   ```
   - Press `i` to open in iOS Simulator (macOS only).
   - Press `a` to open in Android Emulator.
   - Scan the QR code with your phone (using Expo Go) to run on device.

## Building for Production (Native Binaries)

To generate standalone `.apk` (Android) or `.ipa` (iOS) files:

1. **Install EAS CLI**:
   ```bash
   npm install -g eas-cli
   ```

2. **Configure Build**:
   ```bash
   eas build:configure
   ```

3. **Build**:
   - Android: `eas build -p android --profile preview`
   - iOS: `eas build -p ios --profile preview`

This will utilize Expo Application Services (EAS) to compile the native code in the cloud.
