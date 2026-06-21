# CNN_MLP_Snake_AI
PPO deep reinforcement neural network designed and trained to play snake at a high level
Using an actor-critic architecture, it learned to play snake through pure policy to achieve an average length of 60 while sometimes peaking to over 100 on a 20x20 board.

<svg width="100%" viewBox="0 0 680 720" role="img" xmlns="http://www.w3.org/2000/svg">
<title>Snake AI neural network architecture diagram</title>
<desc>Structural diagram showing the CNN feature extractor feeding into actor and critic heads for PPO-based reinforcement learning</desc>
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M2 1L8 5L2 9" fill="none" stroke="#888" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </marker>
  <style>
    text { font-family: sans-serif; fill: #222; }
    .th  { font-size: 14px; font-weight: 500; fill: #222; }
    .ts  { font-size: 12px; fill: #555; }
    .arr { stroke: #888; stroke-width: 1; fill: none; }
  </style>
</defs>

<!-- Input label -->
<text class="ts" x="340" y="28" text-anchor="middle">Input state (3 × 20 × 20)</text>
<text class="ts" x="340" y="44" text-anchor="middle">snake board · head board · food board</text>
<line x1="340" y1="50" x2="340" y2="72" stroke="#888" stroke-width="1" marker-end="url(#arrow)" fill="none"/>

<!-- Feature extractor container -->
<rect x="60" y="76" width="560" height="310" rx="14" fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5"/>
<text class="th" x="340" y="100" text-anchor="middle">Feature extractor (shared)</text>

<!-- Conv1 -->
<rect x="130" y="114" width="420" height="52" rx="8" fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5"/>
<text class="th" x="340" y="137" text-anchor="middle" dominant-baseline="central">Conv2D — depth 8, kernel 3×3</text>
<text class="ts" x="340" y="155" text-anchor="middle" dominant-baseline="central">20×20 → 18×18 × 8   ·   leaky ReLU</text>
<line x1="340" y1="166" x2="340" y2="182" stroke="#888" stroke-width="1" marker-end="url(#arrow)" fill="none"/>

<!-- MaxPool -->
<rect x="200" y="182" width="280" height="44" rx="8" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
<text class="th" x="340" y="200" text-anchor="middle" dominant-baseline="central" style="fill:#085041">Max pool — 2×2</text>
<text class="ts" x="340" y="216" text-anchor="middle" dominant-baseline="central" style="fill:#0F6E56">18×18 → 9×9 × 8</text>
<line x1="340" y1="226" x2="340" y2="242" stroke="#888" stroke-width="1" marker-end="url(#arrow)" fill="none"/>

<!-- Conv2 -->
<rect x="130" y="242" width="420" height="52" rx="8" fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5"/>
<text class="th" x="340" y="264" text-anchor="middle" dominant-baseline="central">Conv2D — depth 16, kernel 3×3</text>
<text class="ts" x="340" y="282" text-anchor="middle" dominant-baseline="central">9×9 → 7×7 × 16   ·   leaky ReLU</text>
<line x1="340" y1="294" x2="340" y2="308" stroke="#888" stroke-width="1" marker-end="url(#arrow)" fill="none"/>

<!-- Reshape + concat -->
<rect x="100" y="308" width="480" height="52" rx="8" fill="#F1EFE8" stroke="#888780" stroke-width="0.5"/>
<text class="th" x="340" y="330" text-anchor="middle" dominant-baseline="central">Flatten + concat extras</text>
<text class="ts" x="340" y="348" text-anchor="middle" dominant-baseline="central">784 flat + direction (4) + length, dx, dy, running → 792</text>
<line x1="340" y1="360" x2="340" y2="374" stroke="#888" stroke-width="1" marker-end="url(#arrow)" fill="none"/>

<!-- Dense shared -->
<rect x="180" y="374" width="320" height="44" rx="8" fill="#EEEDFE" stroke="#7F77DD" stroke-width="0.5"/>
<text class="th" x="340" y="392" text-anchor="middle" dominant-baseline="central">Dense 792 → 64</text>
<text class="ts" x="340" y="408" text-anchor="middle" dominant-baseline="central">leaky ReLU</text>

<!-- Split lines -->
<line x1="340" y1="418" x2="340" y2="438" stroke="#aaa" stroke-width="1" fill="none"/>
<line x1="340" y1="438" x2="175" y2="438" stroke="#aaa" stroke-width="1" fill="none"/>
<line x1="340" y1="438" x2="505" y2="438" stroke="#aaa" stroke-width="1" fill="none"/>
<line x1="175" y1="438" x2="175" y2="458" stroke="#aaa" stroke-width="1" marker-end="url(#arrow)" fill="none"/>
<line x1="505" y1="438" x2="505" y2="458" stroke="#aaa" stroke-width="1" marker-end="url(#arrow)" fill="none"/>

<!-- Actor head container -->
<rect x="60" y="458" width="230" height="190" rx="12" fill="#FAECE7" stroke="#D85A30" stroke-width="0.5"/>
<text class="th" x="175" y="480" text-anchor="middle" style="fill:#4A1B0C">Actor head</text>

<!-- Actor layers -->
<rect x="88" y="492" width="174" height="44" rx="6" fill="#FAECE7" stroke="#D85A30" stroke-width="0.5"/>
<text class="th" x="175" y="510" text-anchor="middle" dominant-baseline="central" style="fill:#4A1B0C">Dense 64 → 32</text>
<text class="ts" x="175" y="526" text-anchor="middle" dominant-baseline="central" style="fill:#712B13">leaky ReLU</text>
<line x1="175" y1="536" x2="175" y2="550" stroke="#D85A30" stroke-width="1" marker-end="url(#arrow)" fill="none"/>

<rect x="88" y="550" width="174" height="44" rx="6" fill="#FAECE7" stroke="#D85A30" stroke-width="0.5"/>
<text class="th" x="175" y="568" text-anchor="middle" dominant-baseline="central" style="fill:#4A1B0C">Dense 32 → 16</text>
<text class="ts" x="175" y="584" text-anchor="middle" dominant-baseline="central" style="fill:#712B13">leaky ReLU</text>
<line x1="175" y1="594" x2="175" y2="608" stroke="#D85A30" stroke-width="1" marker-end="url(#arrow)" fill="none"/>

<rect x="88" y="608" width="174" height="28" rx="6" fill="#FAECE7" stroke="#D85A30" stroke-width="0.5"/>
<text class="th" x="175" y="622" text-anchor="middle" dominant-baseline="central" style="fill:#4A1B0C">Dense 16 → 4   ·   softmax</text>

<!-- Critic head container -->
<rect x="390" y="458" width="230" height="190" rx="12" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
<text class="th" x="505" y="480" text-anchor="middle" style="fill:#04342C">Critic head</text>

<!-- Critic layers -->
<rect x="418" y="492" width="174" height="44" rx="6" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
<text class="th" x="505" y="510" text-anchor="middle" dominant-baseline="central" style="fill:#04342C">Dense 64 → 32</text>
<text class="ts" x="505" y="526" text-anchor="middle" dominant-baseline="central" style="fill:#085041">leaky ReLU</text>
<line x1="505" y1="536" x2="505" y2="574" stroke="#1D9E75" stroke-width="1" marker-end="url(#arrow)" fill="none"/>

<rect x="418" y="574" width="174" height="44" rx="6" fill="#E1F5EE" stroke="#1D9E75" stroke-width="0.5"/>
<text class="th" x="505" y="594" text-anchor="middle" dominant-baseline="central" style="fill:#04342C">Dense 32 → 1</text>
<text class="ts" x="505" y="610" text-anchor="middle" dominant-baseline="central" style="fill:#085041">linear (state value V)</text>

<!-- Output labels -->
<text class="ts" x="175" y="658" text-anchor="middle">π(a|s) — action probs</text>
<text class="ts" x="505" y="658" text-anchor="middle">V(s) — state value</text>

<!-- PPO label -->
<text class="ts" x="340" y="700" text-anchor="middle" font-style="italic">trained with proximal policy optimisation (PPO)</text>
</svg>
