import { create } from 'zustand';

export type UserRole = 'customer' | 'org' | null;

interface User {
  uid: string;
  email: string;
  displayName: string;
  photoURL: string | null;
}

interface AuthStore {
  user: User | null;
  role: UserRole;
  isAuthenticated: boolean;
  setUser: (user: User | null) => void;
  setRole: (role: UserRole) => void;
  logout: () => void;
  initializeAuth: () => void;
}

export const useAuthStore = create<AuthStore>((set) => {
  const initializeAuth = () => {
    const storedRole = localStorage.getItem('userRole') as UserRole;
    if (storedRole) {
      set({ role: storedRole });
    }
  };

  return {
    user: null,
    role: null,
    isAuthenticated: false,
    setUser: (user) =>
      set({
        user,
        isAuthenticated: !!user,
      }),
    setRole: (role) => {
      set({ role });
      if (role) {
        localStorage.setItem('userRole', role);
      } else {
        localStorage.removeItem('userRole');
      }
    },
    logout: () => {
      set({ user: null, role: null, isAuthenticated: false });
      localStorage.removeItem('userRole');
    },
    initializeAuth,
  };
});
