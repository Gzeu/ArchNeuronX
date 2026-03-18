const express = require('express');
const { createProxyMiddleware } = require('http-proxy-middleware');
const WebSocket = require('ws');
const cors = require('cors');
const rateLimit = require('express-rate-limit');
const path = require('path');

class ArchNeuronXWebGateway {
    constructor() {
        this.app = express();
        this.port = process.env.PORT || 3000;
        this.archneuronxApiUrl = process.env.ARCHNEURONX_API_URL || 'http://localhost:8080';
        this.setupMiddleware();
        this.setupRoutes();
        this.setupWebSocket();
        this.setupStaticFiles();
    }
    
    setupMiddleware() {
        // Rate limiting
        const limiter = rateLimit({
            windowMs: 15 * 60 * 1000, // 15 minutes
            max: 1000, // limit each IP to 1000 requests per windowMs
            message: {
                error: 'Too many requests from this IP, please try again later.'
            },
            standardHeaders: true,
            legacyHeaders: false,
        });
        
        this.app.use(limiter);
        this.app.use(cors());
        this.app.use(express.json());
        this.app.use(express.urlencoded({ extended: true }));
    }
    
    setupRoutes() {
        // Health check
        this.app.get('/health', (req, res) => {
            res.json({
                status: 'healthy',
                service: 'ArchNeuronX Web Gateway',
                version: '4.0.0',
                timestamp: new Date().toISOString(),
                uptime: process.uptime()
            });
        });
        
        // Proxy API requests to ArchNeuronX v4.0
        this.app.use('/api/v4', createProxyMiddleware({
            target: this.archneuronxApiUrl,
            changeOrigin: true,
            pathRewrite: {
                '^/api/v4': '/api/v4'
            },
            onProxyReq: (proxyReq, req, res) => {
                // Add custom headers
                proxyReq.setHeader('X-Web-Gateway', 'ArchNeuronX-v4.0');
                proxyReq.setHeader('X-Request-ID', this.generateRequestId());
            },
            onProxyRes: (proxyRes, req, res) => {
                // Log response
                console.log(`[${new Date().toISOString()}] ${req.method} ${req.path} -> ${proxyRes.statusCode}`);
            },
            onError: (err, req, res) => {
                console.error('Proxy error:', err);
                res.status(500).json({
                    error: 'Proxy Error',
                    message: 'Failed to connect to ArchNeuronX API'
                });
            }
        }));
        
        // Enhanced API endpoints for web interface
        this.app.get('/api/v4/dashboard/overview', async (req, res) => {
            try {
                // Fetch data from multiple endpoints
                const [status, models, signals, portfolio] = await Promise.all([
                    this.fetchFromAPI('/api/v4/status'),
                    this.fetchFromAPI('/api/v4/models'),
                    this.fetchFromAPI('/api/v4/signals'),
                    this.fetchFromAPI('/api/v4/portfolio')
                ]);
                
                res.json({
                    status: status.status,
                    performance: status.performance,
                    models: models.models,
                    signals: signals.signals,
                    portfolio: portfolio.portfolio,
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                res.status(500).json({
                    error: 'Failed to fetch dashboard data',
                    message: error.message
                });
            }
        });
        
        // Real-time streaming endpoint
        this.app.get('/api/v4/stream', (req, res) => {
            res.writeHead(200, {
                'Content-Type': 'text/event-stream',
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            });
            
            // Send real-time updates
            const interval = setInterval(async () => {
                try {
                    const status = await this.fetchFromAPI('/api/v4/status');
                    res.write(`data: ${JSON.stringify(status)}\n\n`);
                } catch (error) {
                    console.error('Stream error:', error);
                }
            }, 1000);
            
            req.on('close', () => {
                clearInterval(interval);
            });
        });
        
        // Fallback route for SPA
        this.app.get('*', (req, res) => {
            res.sendFile(path.join(__dirname, '../v4_dashboard/index.html'));
        });
    }
    
    setupWebSocket() {
        const wss = new WebSocket.Server({ port: this.port + 1 });
        
        console.log(`WebSocket server listening on port ${this.port + 1}`);
        
        wss.on('connection', (ws, req) => {
            console.log('New WebSocket connection established');
            
            // Send initial data
            this.sendInitialData(ws);
            
            // Handle incoming messages
            ws.on('message', async (message) => {
                try {
                    const data = JSON.parse(message);
                    await this.handleWebSocketMessage(ws, data);
                } catch (error) {
                    console.error('WebSocket message error:', error);
                    ws.send(JSON.stringify({
                        type: 'error',
                        message: 'Invalid message format'
                    }));
                }
            });
            
            // Handle connection close
            ws.on('close', () => {
                console.log('WebSocket connection closed');
            });
            
            // Handle errors
            ws.on('error', (error) => {
                console.error('WebSocket error:', error);
            });
        });
    }
    
    setupStaticFiles() {
        // Serve static files from v4_dashboard
        this.app.use(express.static(path.join(__dirname, '../v4_dashboard')));
    }
    
    async fetchFromAPI(endpoint) {
        const response = await fetch(`${this.archneuronxApiUrl}${endpoint}`);
        if (!response.ok) {
            throw new Error(`API request failed: ${response.status}`);
        }
        return response.json();
    }
    
    async handleWebSocketMessage(ws, data) {
        switch (data.type) {
            case 'subscribe':
                await this.handleSubscription(ws, data);
                break;
            case 'unsubscribe':
                await this.handleUnsubscription(ws, data);
                break;
            case 'ping':
                ws.send(JSON.stringify({ type: 'pong', timestamp: Date.now() }));
                break;
            default:
                ws.send(JSON.stringify({
                    type: 'error',
                    message: `Unknown message type: ${data.type}`
                }));
        }
    }
    
    async handleSubscription(ws, data) {
        const { channel } = data;
        
        // Add to subscription list
        if (!ws.subscriptions) {
            ws.subscriptions = new Set();
        }
        ws.subscriptions.add(channel);
        
        ws.send(JSON.stringify({
            type: 'subscribed',
            channel: channel,
            timestamp: Date.now()
        }));
        
        // Start sending real-time data for this channel
        this.startRealTimeUpdates(ws, channel);
    }
    
    async handleUnsubscription(ws, data) {
        const { channel } = data;
        
        if (ws.subscriptions) {
            ws.subscriptions.delete(channel);
        }
        
        ws.send(JSON.stringify({
            type: 'unsubscribed',
            channel: channel,
            timestamp: Date.now()
        }));
    }
    
    async sendInitialData(ws) {
        try {
            const status = await this.fetchFromAPI('/api/v4/status');
            ws.send(JSON.stringify({
                type: 'initial_data',
                data: status,
                timestamp: Date.now()
            }));
        } catch (error) {
            console.error('Failed to send initial data:', error);
        }
    }
    
    startRealTimeUpdates(ws, channel) {
        const interval = setInterval(async () => {
            if (ws.readyState === WebSocket.OPEN) {
                try {
                    let data;
                    switch (channel) {
                        case 'status':
                            data = await this.fetchFromAPI('/api/v4/status');
                            break;
                        case 'signals':
                            data = await this.fetchFromAPI('/api/v4/signals');
                            break;
                        case 'portfolio':
                            data = await this.fetchFromAPI('/api/v4/portfolio');
                            break;
                        default:
                            return;
                    }
                    
                    ws.send(JSON.stringify({
                        type: 'update',
                        channel: channel,
                        data: data,
                        timestamp: Date.now()
                    }));
                } catch (error) {
                    console.error(`Failed to fetch ${channel} data:`, error);
                }
            } else {
                clearInterval(interval);
            }
        }, 1000);
        
        // Store interval reference for cleanup
        if (!ws.intervals) {
            ws.intervals = new Map();
        }
        ws.intervals.set(channel, interval);
    }
    
    generateRequestId() {
        return Math.random().toString(36).substring(2, 15);
    }
    
    start() {
        this.app.listen(this.port, () => {
            console.log(`🚀 ArchNeuronX Web Gateway v4.0`);
            console.log(`📡 HTTP Server: http://localhost:${this.port}`);
            console.log(`🔌 WebSocket Server: ws://localhost:${this.port + 1}`);
            console.log(`🎯 ArchNeuronX API: ${this.archneuronxApiUrl}`);
            console.log(`⚡ Ready to serve quantum neural trading dashboard!`);
        });
    }
}

// Start the gateway
const gateway = new ArchNeuronXWebGateway();
gateway.start();

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\n🛑 Shutting down ArchNeuronX Web Gateway...');
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('\n🛑 Shutting down ArchNeuronX Web Gateway...');
    process.exit(0);
});
