# 7a) Vertical RSI‐proportional shading on main chart
#    merge contiguous RSI>50 into spans, opacity ∝ mean RSI in span
max_opacity = 0.3
mask = rsi > 50

spans = []
in_span = False
for dt, is_hot in zip(rsi.index, mask):
    if is_hot and not in_span:
        start = dt
        in_span = True
    elif not is_hot and in_span:
        end = prev_dt + timedelta(days=1)
        span_rsi = rsi[start:prev_dt].mean()
        spans.append((start, end, span_rsi))
        in_span = False
    prev_dt = dt

if in_span:
    end = prev_dt + timedelta(days=1)
    span_rsi = rsi[start:prev_dt].mean()
    spans.append((start, end, span_rsi))

for start, end, span_rsi in spans:
    # compute opacity: 0 at RSI=50, max_opacity at RSI=100
    op = ((span_rsi - 50) / 50) * max_opacity
    fig.add_vrect(
        x0=start, x1=end,
        y0=0, y1=1,
        fillcolor="red",
        opacity=op,
        layer="below",
        row=1, col=1,
        yref="paper"
    )


