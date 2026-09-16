/**
 * BlinkFlow 60FPS Waveform Telemetry Chart
 * Real-time dual-eye openness waveform with animated threshold guides.
 */

class WaveformChart {
  constructor(canvasId, maxPoints = 140) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.maxPoints = maxPoints;

    this.leftData = new Array(maxPoints).fill(0);
    this.rightData = new Array(maxPoints).fill(0);
    this.threshold = 0.55;

    this._setupDpr();
    window.addEventListener('resize', () => this._setupDpr());
  }

  _setupDpr() {
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    this.width = rect.width;
    this.height = rect.height;
    this.canvas.width = this.width * dpr;
    this.canvas.height = this.height * dpr;
    this.ctx.scale(dpr, dpr);
  }

  setThreshold(val) {
    this.threshold = val;
  }

  pushSample(leftScore, rightScore) {
    this.leftData.shift();
    this.leftData.push(leftScore);
    this.rightData.shift();
    this.rightData.push(rightScore);
    this.render();
  }

  render() {
    if (!this.ctx) return;
    const w = this.width;
    const h = this.height;

    this.ctx.clearRect(0, 0, w, h);

    // Draw Subtle Grid
    this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    this.ctx.lineWidth = 1;
    
    // Horizontal 50% line
    this.ctx.beginPath();
    this.ctx.moveTo(0, h * 0.5);
    this.ctx.lineTo(w, h * 0.5);
    this.ctx.stroke();

    // Draw Threshold Guide (Dashed Amber)
    const threshY = h * (1.0 - this.threshold);
    this.ctx.save();
    this.ctx.setLineDash([4, 4]);
    this.ctx.strokeStyle = 'rgba(255, 184, 0, 0.7)';
    this.ctx.lineWidth = 1.5;
    this.ctx.shadowColor = 'rgba(255, 184, 0, 0.5)';
    this.ctx.shadowBlur = 4;
    this.ctx.beginPath();
    this.ctx.moveTo(0, threshY);
    this.ctx.lineTo(w, threshY);
    this.ctx.stroke();
    this.ctx.restore();

    const step = w / (this.maxPoints - 1);

    // Draw Right Eye Wave (Magenta)
    this._drawCurve(this.rightData, '#bd00ff', 'rgba(189, 0, 255, 0.15)', step, h);

    // Draw Left Eye Wave (Cyan)
    this._drawCurve(this.leftData, '#00e5ff', 'rgba(0, 229, 255, 0.15)', step, h);
  }

  _drawCurve(data, strokeColor, fillColor, step, h) {
    this.ctx.save();
    this.ctx.beginPath();
    
    for (let i = 0; i < data.length; i++) {
      const x = i * step;
      // Invert score: 0 is at bottom (h), 1 is at top (0)
      const val = Math.max(0, Math.min(1, data[i]));
      const y = h - (val * (h - 8)) - 4;

      if (i === 0) {
        this.ctx.moveTo(x, y);
      } else {
        this.ctx.lineTo(x, y);
      }
    }

    this.ctx.strokeStyle = strokeColor;
    this.ctx.lineWidth = 2;
    this.ctx.shadowColor = strokeColor;
    this.ctx.shadowBlur = 6;
    this.ctx.stroke();

    // Area fill
    this.ctx.lineTo(step * (data.length - 1), h);
    this.ctx.lineTo(0, h);
    this.ctx.closePath();
    this.ctx.fillStyle = fillColor;
    this.ctx.fill();

    this.ctx.restore();
  }
}

window.WaveformChart = WaveformChart;
