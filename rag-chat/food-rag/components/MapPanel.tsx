"use client";

import { MapContainer, TileLayer, Marker, Popup, Tooltip, useMap } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet";
import { useEffect } from "react";

// Fix Leaflet's default icon path issues in Next.js
const icon = L.icon({
    iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
    iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
    shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
    iconSize: [25, 41],
    iconAnchor: [12, 41],
    popupAnchor: [1, -34],
    tooltipAnchor: [16, -28], // Adjusts where the text appears relative to the pin
});

// Helper to auto-zoom to fit all markers
function MapBounds({ markers }: { markers: { lat: number; lon: number }[] }) {
    const map = useMap();
    useEffect(() => {
        if (markers.length > 0) {
            const bounds = L.latLngBounds(markers.map((m) => [m.lat, m.lon]));
            map.fitBounds(bounds, { padding: [50, 50], maxZoom: 15 });
        }
    }, [markers, map]);
    return null;
}

type Source = {
    name: string;
    address: string;
    lat?: number;
    lon?: number;
};

export default function MapPanel({ sources }: { sources: Source[] }) {
    // Filter for valid coordinates
    const markers = sources.filter((s) => s.lat && s.lon) as (Source & { lat: number; lon: number })[];

    // Default center (Chicago) if no markers
    const defaultCenter: [number, number] = [41.8781, -87.6298];

    return (
        <div className="w-full h-full bg-[#151515] relative z-0">
            {/* Custom CSS to style the tooltips seamlessly on the dark map */}
            <style jsx global>{`
                .glass-tooltip {
                    background: rgba(0, 0, 0, 0.7) !important;
                    border: 1px solid #333 !important;
                    color: white !important;
                    font-weight: 600;
                    font-size: 12px;
                    border-radius: 6px;
                    padding: 2px 6px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                }
                /* Hide the little triangle pointer of the tooltip for a cleaner look */
                .leaflet-tooltip-bottom:before {
                    border-bottom-color: rgba(0, 0, 0, 0.7) !important;
                }
            `}</style>

            <MapContainer
                center={defaultCenter}
                zoom={12}
                style={{ height: "100%", width: "100%", background: "#151515" }}
                zoomControl={false}
            >
                <TileLayer
                    attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                    url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
                />

                {markers.map((s, idx) => (
                    <Marker key={idx} position={[s.lat, s.lon]} icon={icon}>
                        {/* 1. PERMANENT TOOLTIP: Shows text always */}
                        <Tooltip
                            permanent
                            direction="bottom"
                            offset={[0, 10]}
                            className="glass-tooltip"
                        >
                            {s.name}
                        </Tooltip>

                        {/* 2. POPUP: Shows details on click */}
                        <Popup className="custom-popup">
                            <div className="p-1">
                                <h3 className="font-bold text-sm">{s.name}</h3>
                                <p className="text-xs text-gray-500">{s.address}</p>
                            </div>
                        </Popup>
                    </Marker>
                ))}

                <MapBounds markers={markers} />
            </MapContainer>
        </div>
    );
}