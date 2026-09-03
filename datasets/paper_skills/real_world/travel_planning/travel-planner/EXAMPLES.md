# Travel Planner Usage Examples

This document provides real-world examples of using the travel-planner skill for various trip types and preferences.

## Table of Contents

- [Basic Usage](#basic-usage)
- [Weekend Getaways](#weekend-getaways)
- [Week-Long Vacations](#week-long-vacations)
- [Food-Focused Trips](#food-focused-trips)
- [Family Travel](#family-travel)
- [Budget Travel](#budget-travel)
- [Cultural Immersion](#cultural-immersion)
- [Multi-City Tours](#multi-city-tours)
- [Adventure Travel](#adventure-travel)

## Basic Usage

### Example 1: Simple 3-Day City Trip

**Input:**
```
Plan a 3-day trip to Barcelona in September
```

**What Happens:**

1. **Input Parsing**
   - Destination: Barcelona, Spain
   - Duration: 3 days
   - Month: September
   - Preferences: None specified (general sightseeing)

2. **Agent Deployment** (parallel)
   - trip-architect researches: Sagrada Familia, Park Güell, Gothic Quarter, La Rambla, beaches
   - food-explorer finds: Tapas bars, paella restaurants, Boqueria market, wine bars
   - logistics-planner checks: Metro routes, hotel locations, airport transfers
   - local-intel gathers: Pickpocket warnings, siesta hours, Catalan phrases
   - budget-calculator estimates: €80-120/day (accommodation, food, attractions)

3. **Output Generated** in `TRAVEL/barcelona/`
   ```
   ├── README.md                    # Best September weather, crowds moderate
   ├── itinerary.md                 # Day 1: Gothic Quarter + Rambla
   │                                # Day 2: Sagrada Familia + Park Güell
   │                                # Day 3: Beach + Barceloneta
   ├── food_guide.md                # 15 tapas recommendations, market guide
   ├── hotel_recommendations.md     # 3 options: Gothic, Eixample, Barceloneta
   ├── transportation.md            # T-10 ticket, metro map, airport bus
   ├── safety_and_tips.md          # Pickpocket hotspots, scam warnings
   ├── budget_summary.md           # €240-360 total estimate
   └── quick_reference.md          # Emergency: 112, basic Catalan phrases
   ```

---

## Weekend Getaways

### Example 2: Quick 2-Day Escape

**Input:**
```
I have a free weekend next month and want to visit Portland, Oregon. I love coffee and hiking.
```

**Customization Applied:**
- Duration detected: 2 days
- Interests parsed: coffee culture, hiking trails
- trip-architect prioritizes: Forest Park trails, Pittock Mansion hike, Powell's Books
- food-explorer emphasizes: Third-wave coffee shops, food carts, breweries
- logistics-planner includes: Trail parking, coffee shop walking routes

**Sample Output Excerpt** (`itinerary.md`):

```markdown
## Day 1: Urban Exploration + Coffee Culture

### Morning (9:00 AM - 12:00 PM)
**Stumptown Coffee Roasters** (Downtown)
- Why: Iconic Portland coffee, tour available
- Duration: 1 hour
- Cost: $15-20

**Powell's City of Books** (Pearl District)
- Why: World's largest independent bookstore, coffee bar inside
- Duration: 2 hours
- Walk: 15 minutes from Stumptown

### Afternoon (12:00 PM - 6:00 PM)
**Forest Park - Wildwood Trail**
- Why: Urban wilderness, 5-mile loop with city views
- Trailhead: NW Thurman St (parking available)
- Difficulty: Moderate
- Duration: 3-4 hours including drive
...
```

---

## Week-Long Vacations

### Example 3: Comprehensive 7-Day Journey

**Input:**
```
Plan a week in Japan - arriving in Tokyo, want to see temples and traditional culture
```

**Complex Planning:**
- Multi-city coordination (Tokyo → Kyoto → Nara)
- JR Pass recommendation by logistics-planner
- Temple circuit design by trip-architect
- Traditional kaiseki dining by food-explorer
- Onsen etiquette by local-intel

**Output Structure:**
```
TRAVEL/japan-tokyo-kyoto/
├── README.md                    # Overview: Tokyo (3d) → Kyoto (3d) → Nara (1d)
├── itinerary.md                 # 7-day schedule with travel days
├── food_guide.md                # Kaiseki, ramen, sushi by city
├── hotel_recommendations.md     # Tokyo hotel + Kyoto ryokan
├── transportation.md            # JR Pass guide, Shinkansen reservations
├── safety_and_tips.md          # Temple etiquette, onsen rules, gift-giving
├── budget_summary.md           # $2000-3000 (JR Pass, hotels, food)
└── quick_reference.md          # Useful phrases, WiFi rental, IC cards
```

**Sample Day** (`itinerary.md`):
```markdown
## Day 4: Kyoto - Eastern Temples

### Early Morning (6:30 AM - 9:00 AM)
**Fushimi Inari Shrine** 🦊
- Why: Iconic vermillion torii gates, fewer crowds at dawn
- Access: Inari Station (JR Nara Line, 5 min from Kyoto Station)
- Visit Duration: 2-3 hours (full hike to summit)
- Cost: Free
- Tip: Start early to avoid tour groups, bring water

### Late Morning (10:00 AM - 12:00 PM)
**Tofuku-ji Temple** (東福寺)
- Why: Stunning Zen garden, maple bridges (famous in autumn)
- Access: 1 station from Inari (JR)
- Duration: 1.5 hours
- Cost: ¥600
...
```

---

## Food-Focused Trips

### Example 4: Culinary Adventure

**Input:**
```
4 days in New Orleans, I'm a foodie and want authentic Creole cuisine
```

**food-explorer Takes Lead:**
- Prioritizes James Beard Award restaurants
- Includes cooking class recommendations
- Maps food tour routes (French Quarter, Garden District)
- Historical context for Creole vs Cajun

**Sample Output** (`food_guide.md`):

```markdown
## Must-Try Dishes

### Gumbo
**What It Is:** Rich stew with okra, roux, seafood/sausage, holy trinity (celery, bell pepper, onion)

**Best Places:**
1. **Dooky Chase's Restaurant** (Treme)
   - Leah Chase's legendary chicken and sausage gumbo
   - Lunch only, $18-25
   - Reservations recommended

2. **Gumbo Shop** (French Quarter)
   - Seafood gumbo with crab, shrimp, oysters
   - $12-18, casual atmosphere
   - No reservations needed

### Beignets
**What They Are:** Fried dough pillows covered in powdered sugar

**Café Du Monde** (French Quarter) - The Original
- 24/7 operation
- $3 for 3 beignets + café au lait
- Cash only, outdoor seating
- Expect 20-30 min wait during peak hours
...

## Food Tours & Experiences

### Cooking Classes
**New Orleans School of Cooking** (French Quarter)
- Learn to make gumbo, jambalaya, pralines
- 2-hour demonstration class: $35
- Hands-on class: $125
- Book 2 weeks ahead
...
```

---

## Family Travel

### Example 5: Kid-Friendly Vacation

**Input:**
```
5-day San Diego trip with kids ages 6 and 9, late July
```

**Adaptations:**
- trip-architect prioritizes: Zoo, beaches, LEGOLAND, USS Midway
- food-explorer includes: Kid-friendly restaurants, picnic spots
- local-intel adds: Sunscreen reminder, beach safety, child ID bracelets
- budget-calculator accounts for: Family passes, larger hotel room

**Sample Output** (`itinerary.md`):

```markdown
## Day 2: LEGOLAND California

### Morning (9:00 AM - 12:00 PM)
**LEGOLAND Theme Park** (Carlsbad, 35 min drive)

**Arrival Strategy:**
- Arrive at 9 AM (opening) to avoid crowds
- Head straight to Ninjago World (most popular)
- Use Rider Switch if kids have different height requirements

**Must-Do Attractions:**
- Ninjago The Ride (interactive, ages 6+)
- LEGO Factory Tour (educational, air-conditioned break)
- Miniland USA (impressive LEGO replicas)

**Tips:**
- Bring sunscreen and hats (limited shade)
- Refillable water bottles allowed
- Pack snacks (food expensive inside)

### Afternoon (12:00 PM - 3:00 PM)
**Lunch at Fun Town Market** (inside park)
- Pizza, chicken tenders, salads
- Budget: $50-70 for family of 4

**SEA LIFE Aquarium** (attached to LEGOLAND)
- Touch pools (kids love ray feeding)
- 360° ocean tunnel
- Duration: 1.5 hours
...
```

---

## Budget Travel

### Example 6: Affordable Adventure

**Input:**
```
Plan a cheap 4-day trip to Lisbon, budget is tight
```

**budget-calculator Emphasizes:**
- Free attractions (viewpoints, street art)
- Grocery shopping vs restaurants
- Public transit optimization
- Hostel vs Airbnb cost comparison

**Sample Output** (`budget_summary.md`):

```markdown
## Daily Budget Breakdown (Low-Cost Option)

### Accommodation: €25-35/night
**Recommended:** Hostels in Baixa/Chiado
- Home Lisbon Hostel: €30/night (private room)
- Includes breakfast, kitchen access
- **4 nights = €120**

### Transportation: €6.50/day
- **7-day metro/tram pass:** €26 (covers all 4 days)
- Airport bus: €4 one-way
- **Total = €34**

### Food: €15-20/day
**Breakfast:** Included at hostel (€0)
**Lunch:**
- Pastelaria takeaway: €5-7 (sandwich + pastéis de nata)
- Supermarket salad: €3-4
**Dinner:**
- Tasca (neighborhood tavern): €8-12
- Grocery cooking: €5-8
**Total = €60-80**

### Attractions: €5-10/day
**Free:**
- Miradouros (viewpoints): €0
- São Jorge Castle grounds: €0 (interior €10)
- LX Factory street art: €0
- Jerónimos Monastery exterior: €0

**Paid:**
- Tile Museum: €5
- Tram 28 ride (included in pass): €0
- **Total = €20-40**

## 4-Day Total: €234-274 ($250-300)
```

---

## Cultural Immersion

### Example 7: Deep Cultural Experience

**Input:**
```
10-day Morocco trip, interested in history, architecture, and local life
```

**local-intel Provides:**
- Dress code guidance (mosques, medinas)
- Bargaining etiquette for souks
- Ramadan considerations (if applicable)
- Hamman (bathhouse) protocol

**Sample Output** (`safety_and_tips.md`):

```markdown
## Cultural Etiquette

### Mosque Visits
**Dress Code:**
- Women: Cover shoulders, knees, hair (scarf provided at some sites)
- Men: Long pants, covered shoulders
- Remove shoes before entering prayer halls

**Behavior:**
- Ask permission before photographing
- Non-Muslims may not enter some mosques (e.g., during prayer)
- Speak quietly, avoid public displays of affection

### Souk Shopping & Bargaining
**Expected Process:**
1. Show interest in item
2. Ask price (expect 2-3x fair value)
3. Counter at 40-50% of asking
4. Negotiate to middle ground (usually 60-70% of original)

**Tips:**
- Don't bargain if not seriously interested
- Stay friendly, it's a social ritual
- Walk away if price too high (seller may call you back)
- Cash negotiations get better deals than card

### Moroccan Greetings
- Handshakes common between same genders
- "As-salamu alaykum" (Peace be upon you)
- Response: "Wa alaykumu s-salam" (And upon you peace)
- Three cheek kisses between friends (right-left-right)
...
```

---

## Multi-City Tours

### Example 8: European Circuit

**Input:**
```
2 weeks in Europe: Amsterdam → Paris → Rome, relaxed pace
```

**logistics-planner Coordinates:**
- Train booking strategy (Thalys, Trenitalia)
- Luggage storage between hotels
- Day trip options from each city
- Schengen zone entry/exit

**Sample Output** (`transportation.md`):

```markdown
## Inter-City Travel

### Amsterdam → Paris (Day 5)
**Thalys High-Speed Train**
- Duration: 3h 20m direct
- Frequency: Every 1-2 hours
- Cost: €35-150 (book 3 months ahead for cheapest)
- Station: Amsterdam Centraal → Paris Gare du Nord

**Booking Strategy:**
- Use Thalys.com or Trainline.com
- Book at 00:00 exactly 3 months before travel (best prices)
- Select "comfort 2" seats (adequate leg room)
- Luggage: 2 pieces + 1 carry-on included

**Travel Day Tips:**
- Arrive 20 min early (no security, just platform validation)
- Grab stroopwafel at station for train snack
- Metro line 4 from Gare du Nord to hotel (€2.15)

### Paris → Rome (Day 10)
**Flight Recommended** (train = 15+ hours)
- Budget: easyJet, Ryanair (€30-80)
- Standard: Air France, ITA (€100-180)
- Duration: 2h 10m
- Airports: CDG/ORY → FCO

**Airport Transfer:**
- Paris: RER B to CDG (€11.50, 30 min)
- Rome: Leonardo Express to Termini (€14, 32 min)
...
```

---

## Adventure Travel

### Example 9: Active Outdoor Trip

**Input:**
```
1 week in Iceland, want to hike and see waterfalls, July
```

**trip-architect Focuses On:**
- Golden Circle vs Ring Road route
- Hiking difficulty ratings
- Daylight hours (20+ in July)
- Weather backup plans

**Sample Output** (`itinerary.md`):

```markdown
## Day 3: South Coast - Waterfalls & Black Sand Beach

### Morning (8:00 AM - 12:00 PM)
**Seljalandsfoss Waterfall** 🌊
- Why: Walk behind the waterfall, unique perspective
- Drive: 2h from Reykjavik
- Duration: 45 min
- Parking: Free
- **Gear:** Waterproof jacket essential, trail can be slippery

**Gljúfrabúi (Secret Waterfall)**
- 5 min walk from Seljalandsfoss
- Wade through shallow stream to enter canyon
- Duration: 30 min
- **Tip:** Waterproof boots or prepare to get wet feet

### Midday (12:00 PM - 3:00 PM)
**Skógafoss Waterfall**
- Why: Massive 60m drop, climb 400 steps to top viewpoint
- Drive: 30 min east
- Duration: 1.5 hours
- **Hike Option:** Fimmvörðuháls trail (10km, 5-6 hours to next hut)
  - Only for experienced hikers with gear
  - Bring: Map, water, snacks, layers

### Afternoon (3:00 PM - 6:00 PM)
**Reynisfjara Black Sand Beach** 🖤
- Why: Basalt columns, sea stacks, dramatic waves
- Drive: 20 min from Skógafoss
- Duration: 1 hour
- **CRITICAL SAFETY:** Never turn your back on ocean
  - "Sneaker waves" can be fatal
  - Stay well beyond marked danger zone
  - Do not enter caves during high tide

### Evening (7:00 PM - 9:00 PM)
**Dinner in Vík** (village)
- Súper Pizza: Casual, $15-20
- Stay overnight: Vík Hostel or Hotel Kría
...
```

---

## Sample Quick Reference Output

Every plan includes a `quick_reference.md` for at-a-glance info:

```markdown
# Barcelona Quick Reference

## Emergency
- **Emergency:** 112 (all services)
- **Police:** 091
- **Medical:** 061

## Essential Phrases
- Hello: Hola
- Thank you: Gràcies (Catalan) / Gracias (Spanish)
- How much?: Quant costa? / ¿Cuánto cuesta?
- Check please: El compte / La cuenta
- I don't speak Catalan: No parlo català

## Transportation
- **Metro Card:** T-Casual (€12.15, 10 rides)
- **Airport Bus:** Aerobus (€6.75, 35 min)
- **Taxi:** €1.10 flag + €1.20/km
- **Bike Rental:** Bicing (locals), Donkey Republic (tourists)

## Useful Apps
- **Maps:** Citymapper (better than Google for Barcelona)
- **Transit:** TMB (official metro app)
- **Food:** The Fork (restaurant reservations + discounts)
- **Translation:** Google Translate (download Spanish offline)

## Money
- **Currency:** Euro (€)
- **ATMs:** Avoid Euronet (high fees), use bank ATMs
- **Tipping:** Not required, round up 5-10% for good service
- **Card:** Widely accepted, but carry €20 cash

## Weather (September)
- **Temp:** 20-26°C (68-79°F)
- **Rain:** Low chance (~6 days/month)
- **Pack:** Light layers, sunscreen, comfortable shoes

## Hours
- **Restaurants:** Lunch 1-4 PM, Dinner 9-11 PM
- **Shops:** 10 AM-2 PM, 5-8 PM (siesta 2-5 PM)
- **Museums:** Typically 10 AM-8 PM, closed Mondays
```

---

## Using the Skill

To generate any of these plans, simply tell Claude:

```
[Your trip description matching any example above]
```

The skill will automatically:
1. Parse your requirements
2. Deploy 5 specialist agents
3. Research in parallel
4. Synthesize findings
5. Output structured markdown files

**No commands needed - just natural conversation.**
