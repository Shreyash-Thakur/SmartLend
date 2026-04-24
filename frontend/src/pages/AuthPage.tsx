import { useNavigate } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { useEffect, useState } from 'react';
import { firebaseConfigDiagnostics } from '../lib/firebase';

export const AuthPage = () => {
  const navigate = useNavigate();
  const { loginWithGoogle, user, setRole, loading } = useAuth();
  const [showRoleSelection, setShowRoleSelection] = useState(false);
  const [loadingLogin, setLoadingLogin] = useState(false);
  const [loginError, setLoginError] = useState<string | null>(null);

  useEffect(() => {
    if (user && !showRoleSelection) {
      setShowRoleSelection(true);
    }
  }, [user, showRoleSelection]);

  const handleGoogleLogin = async () => {
    setLoginError(null);
    setLoadingLogin(true);
    try {
      await loginWithGoogle();
    } catch (error) {
      console.error('Login failed:', error);
      setLoginError(error instanceof Error ? error.message : 'Unable to sign in with Google.');
      setLoadingLogin(false);
    }
  };

  const handleRoleSelection = (role: 'customer' | 'org') => {
    setRole(role);
    if (role === 'customer') {
      navigate('/dashboard/customer');
    } else {
      navigate('/dashboard/org');
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center">
        <div className="text-center">
          <div className="w-12 h-12 border-4 border-indigo-200 border-t-indigo-600 rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  if (showRoleSelection && user) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
        <div className="bg-white rounded-lg shadow-lg p-8 max-w-md w-full">
          <h2 className="text-2xl font-bold text-center text-gray-900 mb-2">
            Welcome, {user.displayName}!
          </h2>
          <p className="text-center text-gray-600 mb-8">
            How would you like to access SmartLend?
          </p>

          <div className="space-y-4">
            <button
              onClick={() => handleRoleSelection('customer')}
              className="w-full px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg hover:bg-blue-700 transition-colors duration-200"
            >
              Continue as Customer
            </button>
            <button
              onClick={() => handleRoleSelection('org')}
              className="w-full px-6 py-3 bg-indigo-600 text-white font-semibold rounded-lg hover:bg-indigo-700 transition-colors duration-200"
            >
              Continue as Organization
            </button>
          </div>

          <p className="text-sm text-gray-500 text-center mt-6">
            Logged in as: {user.email}
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 flex items-center justify-center p-4">
      <div className="bg-white rounded-lg shadow-lg p-8 max-w-md w-full">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">SmartLend</h1>
          <p className="text-gray-600">
            AI-Powered Loan Decision Platform
          </p>
        </div>

        <button
          onClick={handleGoogleLogin}
          disabled={loadingLogin}
          className="w-full px-6 py-3 bg-white border-2 border-gray-300 text-gray-700 font-semibold rounded-lg hover:bg-gray-50 transition-colors duration-200 flex items-center justify-center gap-2 disabled:opacity-50"
        >
          <svg
            className="w-5 h-5"
            viewBox="0 0 24 24"
            fill="currentColor"
          >
            <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z" />
            <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" />
            <path d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z" />
            <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" />
          </svg>
          {loadingLogin ? 'Signing in...' : 'Sign in with Google'}
        </button>

        {!firebaseConfigDiagnostics.enabled && (
          <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-md p-3 mt-4">
            Firebase configuration incomplete. Missing keys: {firebaseConfigDiagnostics.missingConfigKeys.join(', ')}
          </p>
        )}

        {loginError && (
          <p className="text-xs text-red-700 bg-red-50 border border-red-200 rounded-md p-3 mt-4">
            {loginError}
          </p>
        )}

        <p className="text-sm text-gray-500 text-center mt-6">
          By signing in, you agree to our Terms of Service
        </p>
      </div>
    </div>
  );
};
