import { initializeApp } from 'firebase/app';
import { getAuth, GoogleAuthProvider } from 'firebase/auth';

const firebaseConfig = {
  apiKey: import.meta.env.VITE_FIREBASE_API_KEY,
  authDomain: import.meta.env.VITE_FIREBASE_AUTH_DOMAIN,
  projectId: import.meta.env.VITE_FIREBASE_PROJECT_ID,
  storageBucket: import.meta.env.VITE_FIREBASE_STORAGE_BUCKET,
  messagingSenderId: import.meta.env.VITE_FIREBASE_MESSAGING_SENDER_ID,
  appId: import.meta.env.VITE_FIREBASE_APP_ID,
};

const requiredKeys = [
  'apiKey',
  'authDomain',
  'projectId',
  'storageBucket',
  'messagingSenderId',
  'appId',
] as const;

const missingConfigKeys = requiredKeys.filter((key) => !firebaseConfig[key]);
const isFirebaseConfigured = missingConfigKeys.length === 0;

export const firebaseConfigDiagnostics = {
  enabled: isFirebaseConfigured,
  missingConfigKeys,
};

if (!isFirebaseConfigured) {
  console.warn(
    `[SmartLend] Firebase environment variables are missing (${missingConfigKeys.join(', ')}). `
      + 'Google Auth is disabled until all VITE_FIREBASE_* values are set.',
  );
}

const app = isFirebaseConfigured ? initializeApp(firebaseConfig) : null;

export const auth = app ? getAuth(app) : null;
export const provider = app ? new GoogleAuthProvider() : null;
export const firebaseAuthEnabled = isFirebaseConfigured;

if (provider) {
  provider.setCustomParameters({
    prompt: 'consent',
  });
}
