while true; do
  echo "🔁 Tentative de build avec MAX_JOBS=1..."
  MAX_JOBS=1 python setup.py install && break
  echo "❌ Build échoué. Nouvelle tentative dans 15 secondes..."
  sleep 15
done