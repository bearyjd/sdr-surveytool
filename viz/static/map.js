// Map page script. Survey records carry attacker-influenced strings (SSIDs,
// BLE names): they must only ever reach the DOM as text nodes, never as HTML.
const map = L.map('map').setView([0, 0], 2);
L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
const KEY = 'surveytoken';
const msg = document.getElementById('msg');
const login = document.getElementById('login');

function showLogin(text) {
  msg.textContent = text;
  login.style.display = 'block';
}

function popupFor(record) {
  const box = document.createElement('div');
  const lines = [['modality', record.modality]].concat(
    Object.entries(record.identifier || {})
  );
  lines.forEach(([key, value]) => {
    const line = document.createElement('div');
    line.textContent = String(key) + ': ' + String(value);
    box.appendChild(line);
  });
  return box;
}

function load(token) {
  fetch('/api/records', { headers: { 'Authorization': 'Bearer ' + token } })
    .then(r => {
      if (r.status === 401) {
        sessionStorage.removeItem(KEY);
        showLogin('Invalid or missing token. Enter the API token.');
        return [];
      }
      if (!r.ok) {
        msg.textContent = 'Request failed (' + r.status + ').';
        return [];
      }
      login.style.display = 'none';
      msg.textContent = '';
      return r.json();
    })
    .then(records => {
      records.forEach(r => {
        L.circleMarker([r.lat, r.lon], { radius: 4 })
          .bindPopup(popupFor(r))
          .addTo(map);
      });
    })
    .catch(() => showLogin('Could not load records (network or server error).'));
}

login.addEventListener('submit', e => {
  e.preventDefault();
  const input = document.getElementById('token');
  sessionStorage.setItem(KEY, input.value);
  input.value = '';
  load(sessionStorage.getItem(KEY));
});
const saved = sessionStorage.getItem(KEY);
if (saved) { load(saved); } else { showLogin(''); }
