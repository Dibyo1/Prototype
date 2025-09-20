// TruthGuard AI Service Worker
// Version 1.0.0

const CACHE_NAME = 'truthguard-v1.0.0';
const urlsToCache = [
  '/',
  '/manifest.json',
  'https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/gsap.min.js',
  'https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/ScrollTrigger.min.js',
  'https://cdnjs.cloudflare.com/ajax/libs/gsap/3.12.2/TextPlugin.min.js',
  'https://cdnjs.cloudflare.com/ajax/libs/animejs/3.2.1/anime.min.js',
  'https://cdnjs.cloudflare.com/ajax/libs/particles.js/2.0.0/particles.min.js',
  'https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;500;600;700&display=swap'
];

// Install Service Worker
self.addEventListener('install', (event) => {
  console.log('🛠️ Service Worker installing...');
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('📦 Caching app shell');
        return cache.addAll(urlsToCache);
      })
      .then(() => {
        console.log('✅ Service Worker installed successfully');
        // Force activation of new service worker
        return self.skipWaiting();
      })
      .catch((error) => {
        console.error('❌ Service Worker installation failed:', error);
      })
  );
});

// Activate Service Worker
self.addEventListener('activate', (event) => {
  console.log('🚀 Service Worker activating...');
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName !== CACHE_NAME) {
            console.log('🗑️ Deleting old cache:', cacheName);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      console.log('✅ Service Worker activated');
      // Take control of all clients
      return self.clients.claim();
    })
  );
});

// Fetch Event - Network First with Cache Fallback
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Handle API requests with network-first strategy
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      networkFirstStrategy(request)
    );
    return;
  }
  
  // Handle static assets with cache-first strategy
  if (request.destination === 'script' || 
      request.destination === 'style' || 
      request.destination === 'font' ||
      url.hostname === 'cdnjs.cloudflare.com' ||
      url.hostname === 'fonts.googleapis.com' ||
      url.hostname === 'fonts.gstatic.com') {
    event.respondWith(
      cacheFirstStrategy(request)
    );
    return;
  }
  
  // Handle HTML requests with network-first strategy
  if (request.destination === 'document') {
    event.respondWith(
      networkFirstStrategy(request)
    );
    return;
  }
  
  // Default: cache-first for everything else
  event.respondWith(
    cacheFirstStrategy(request)
  );
});

// Network-first strategy
async function networkFirstStrategy(request) {
  try {
    // Try network first
    const networkResponse = await fetch(request.clone());
    
    // If successful, cache the response and return it
    if (networkResponse.ok) {
      const cache = await caches.open(CACHE_NAME);
      // Only cache GET requests
      if (request.method === 'GET') {
        cache.put(request, networkResponse.clone());
      }
      return networkResponse;
    }
    
    throw new Error('Network response not ok');
  } catch (error) {
    console.log('🌐 Network failed, trying cache for:', request.url);
    
    // Try cache as fallback
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // If it's an API request that failed and no cache, return offline response
    if (request.url.includes('/api/')) {
      return new Response(
        JSON.stringify({
          error: 'You are offline. Please check your internet connection.',
          error_code: 'OFFLINE',
          offline: true
        }),
        {
          status: 503,
          statusText: 'Service Unavailable',
          headers: {
            'Content-Type': 'application/json'
          }
        }
      );
    }
    
    // For HTML requests, return the cached index page
    if (request.destination === 'document') {
      const fallbackResponse = await caches.match('/');
      if (fallbackResponse) {
        return fallbackResponse;
      }
    }
    
    throw error;
  }
}

// Cache-first strategy
async function cacheFirstStrategy(request) {
  try {
    // Try cache first
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
      return cachedResponse;
    }
    
    // If not in cache, fetch from network
    const networkResponse = await fetch(request);
    
    // Cache the response for future use
    if (networkResponse.ok && request.method === 'GET') {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, networkResponse.clone());
    }
    
    return networkResponse;
  } catch (error) {
    console.error('❌ Both cache and network failed for:', request.url);
    throw error;
  }
}

// Background Sync for offline actions
self.addEventListener('sync', (event) => {
  console.log('🔄 Background sync triggered:', event.tag);
  
  if (event.tag === 'truthguard-sync') {
    event.waitUntil(
      syncOfflineActions()
    );
  }
});

// Sync offline actions when connection is restored
async function syncOfflineActions() {
  try {
    const cache = await caches.open(CACHE_NAME);
    const offlineRequests = await cache.match('/offline-queue');
    
    if (offlineRequests) {
      const requests = await offlineRequests.json();
      
      for (const requestData of requests) {
        try {
          await fetch(requestData.url, requestData.options);
          console.log('✅ Synced offline request:', requestData.url);
        } catch (error) {
          console.error('❌ Failed to sync request:', requestData.url, error);
        }
      }
      
      // Clear the offline queue after syncing
      await cache.delete('/offline-queue');
    }
  } catch (error) {
    console.error('❌ Background sync failed:', error);
  }
}

// Push notification handling
self.addEventListener('push', (event) => {
  console.log('📱 Push notification received');
  
  const options = {
    body: 'TruthGuard AI has new verification results!',
    icon: '/icon-192x192.png',
    badge: '/badge-72x72.png',
    vibrate: [100, 50, 100],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    },
    actions: [
      {
        action: 'explore',
        title: 'View Results',
        icon: '/icon-check.png'
      },
      {
        action: 'close',
        title: 'Close',
        icon: '/icon-close.png'
      }
    ]
  };
  
  if (event.data) {
    const pushData = event.data.json();
    options.body = pushData.body || options.body;
    options.data = { ...options.data, ...pushData.data };
  }
  
  event.waitUntil(
    self.registration.showNotification('TruthGuard AI', options)
  );
});

// Notification click handling
self.addEventListener('notificationclick', (event) => {
  console.log('🔔 Notification clicked:', event.action);
  
  event.notification.close();
  
  if (event.action === 'explore') {
    event.waitUntil(
      clients.openWindow('/')
    );
  }
});

// Message handling from main thread
self.addEventListener('message', (event) => {
  console.log('💬 Message received in SW:', event.data);
  
  if (event.data && event.data.type === 'SKIP_WAITING') {
    self.skipWaiting();
  }
  
  if (event.data && event.data.type === 'CACHE_URLS') {
    event.waitUntil(
      caches.open(CACHE_NAME).then((cache) => {
        return cache.addAll(event.data.urls);
      })
    );
  }
});

// Log service worker status
console.log('🛡️ TruthGuard Service Worker loaded');