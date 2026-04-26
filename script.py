import re
with open('frontend/src/pages/Dashboard.models.tsx', 'r') as f:
    c = f.read()

c = c.replace('#2563eb', '#6E61FF')
c = c.replace('#0ea5e9', '#38BDF8')
c = c.replace('#14b8a6', '#B0F0DA')
c = c.replace('#f97316', '#FD9745')
c = c.replace('#8b5cf6', '#FF6B6B')
c = c.replace('#10b981', '#B0F0DA')
c = c.replace('#ef4444', '#FF6B6B')
c = c.replace('#f59e0b', '#FD9745')
c = c.replace('#16a34a', '#B0F0DA')
c = c.replace('strokeDasharray="3 3" stroke="#e5e7eb"', 'strokeDasharray="0" stroke="#000000"')

with open('frontend/src/pages/Dashboard.models.tsx', 'w') as f:
    f.write(c)
