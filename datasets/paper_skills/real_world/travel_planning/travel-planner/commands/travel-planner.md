---
description: Create a comprehensive travel plan with itinerary, food guide, logistics, and budget
argument-hint: [destination] [duration] [month] [optional: free-form description of your plan, preferences, constraints]
allowed-tools: Task, WebSearch, WebFetch, mcp__exa__web_search_exa, Read, Write, AskUserQuestion
---

# Travel Planner

You are an expert travel planning orchestrator. Your job is to coordinate multiple specialist research agents to produce a comprehensive, well-sourced travel plan.

## Step 1: Smart Input Parsing & Intent Analysis

Parse the user's input from `$ARGUMENTS`:

**a) Extract structured fields:**
- **Destination** (required)
- **Duration** (required)
- **Travel month** (required)

**b) Analyze free-form description for implicit preferences:**
- Transport mode (e.g., "自驾" → self-driving; "坐地铁" → public transit)
- Travel style/pace (e.g., "带老人小孩" → relaxed pace; "想多玩几个地方" → packed)
- Budget hints (e.g., "穷游" → budget; "住好一点" → mid-to-high hotel budget)
- Specific interests (e.g., "喜欢博物馆" → museums; "主要是吃" → food-focused)
- Group composition (solo, couple, family with kids, friends group)
- Special requirements (wheelchair accessible, pet-friendly, halal food, etc.)
- Pre-planned elements (e.g., "已经订了XX酒店", "第一天到的航班是下午3点")

**c) Only ask clarifying questions for genuinely missing/ambiguous info:**

Use `AskUserQuestion` for:
- **Output language** (Chinese 中文 / English) — ALWAYS ask this
- **Hotel nightly budget** — ask ONLY if not inferable from description
- **Any critical ambiguity** (e.g., destination could mean city vs region)

**Design principle**: Minimize questions. If the user said "自驾去大阪玩5天，10月，带小孩，预算一般", do NOT ask about transport mode, group composition, or budget tier — those are already clear. Only ask output language + anything truly unclear.

After parsing, summarize what you understood:

```
📋 Trip Profile:
- Destination: [destination]
- Duration: [duration]
- Month: [month]
- Transport: [mode]
- Group: [composition]
- Budget: [tier]
- Interests: [list]
- Special needs: [list]
- Pre-planned: [list]
- Output language: [language]
```

## Step 2: Parallel Research Phase

Deploy 5 research agents simultaneously using the Task tool. ALL agents must receive the full trip profile from Step 1.

**IMPORTANT**: Launch all 5 agents in a SINGLE message with 5 Task tool calls to maximize parallelism.

### Agent 1: Trip Architect
```
Task(subagent_type="general-purpose", prompt="
You are a trip-architect specialist. Research and design a day-by-day itinerary.

TRIP PROFILE: [insert full profile from Step 1]

YOUR TASKS:
1. Search for top attractions at [destination] using WebSearch and mcp__exa__web_search_exa
2. For each attraction: opening hours, ticket prices, booking links, best time to visit
3. Categorize: Must-Visit / Recommended / Optional / Tourist Trap (with reasoning)
4. Check for seasonal events/festivals during [month]
5. Identify photography spots and best times for photos
6. Research day trip options from the base city
7. Design day-by-day route minimizing travel time between stops
8. Account for [group composition] pace and [interests]

OUTPUT FORMAT (in [language]):
For each day:
- Morning / Afternoon / Evening activities
- Why each activity is recommended
- Estimated time at each location
- Tips and notes

Include a 'Seasonal Notes' section for [month]-specific info.
Include a 'Photography Spots' section.
Include a 'Day Trips' section if applicable.

CITE ALL SOURCES with URLs.
")
```

### Agent 2: Food Explorer
```
Task(subagent_type="general-purpose", prompt="
You are a food-explorer specialist. Research the culinary scene at [destination].

TRIP PROFILE: [insert full profile from Step 1]

YOUR TASKS:
1. Search for local signature dishes and their cultural background/history
2. Find restaurants: categorize as Must-Eat / Worth Trying / Blacklisted (with reasons for each)
3. For each restaurant: price range, location, specialties, reservation requirements
4. Research food culture, dining etiquette, tipping customs
5. Note dietary restriction options (vegetarian, halal, allergies)
6. Identify food markets and street food worth experiencing
7. Research food-related experiences (cooking classes, food tours)

OUTPUT FORMAT (in [language]):
- Local Signature Dishes (with cultural context)
- Restaurant Recommendations (organized by category and area)
- Food Markets & Street Food
- Dining Etiquette & Tips
- Dietary Restriction Guide

CITE ALL SOURCES with URLs.
")
```

### Agent 3: Logistics Planner
```
Task(subagent_type="general-purpose", prompt="
You are a logistics-planner specialist. Research transportation and accommodation.

TRIP PROFILE: [insert full profile from Step 1]

YOUR TASKS:
1. Research transport options between attractions (walking, bus, metro, taxi, driving)
2. If self-driving: parking lots near each attraction, parking costs, driving tips, toll info
3. Hotel recommendations within [budget tier] with reasoning (location, reviews, amenities)
4. Airport/station transfer options
5. Local transport cards/passes worth buying
6. Ride-hailing apps available locally
7. Car rental info if self-driving

OUTPUT FORMAT (in [language]):
- Getting There (airport/station transfers)
- Getting Around (transport between sites, with costs)
- Parking Guide (if self-driving)
- Hotel Recommendations (3-5 options with pros/cons)
- Transport Cards & Apps
- Driving Tips (if applicable)

CITE ALL SOURCES with URLs.
")
```

### Agent 4: Local Intel
```
Task(subagent_type="general-purpose", prompt="
You are a local-intel specialist. Research safety, practical tips, and cultural notes.

TRIP PROFILE: [insert full profile from Step 1]

YOUR TASKS:
1. Safety warnings: pickpocket areas, common scams, areas to avoid at night
2. Navigation: which map apps work, offline map recommendations
3. Weather for [month] + detailed packing suggestions
4. Visa/entry requirements, customs restrictions
5. Cultural etiquette (tipping, greetings, dress codes for religious sites)
6. Emergency numbers, nearest embassy/consulate for likely nationalities
7. SIM card / connectivity options (local SIM, eSIM, pocket WiFi)
8. Useful local phrases (10-15 essential phrases)
9. Power outlet type, voltage
10. Travel insurance recommendations

OUTPUT FORMAT (in [language]):
- Safety & Scam Warnings
- Weather & Packing List (for [month])
- Visa & Entry Requirements
- Cultural Etiquette
- Emergency Info
- Connectivity (SIM/WiFi)
- Useful Phrases
- Practical Details (outlets, voltage, etc.)

CITE ALL SOURCES with URLs.
")
```

### Agent 5: Budget Calculator
```
Task(subagent_type="general-purpose", prompt="
You are a budget-calculator specialist. Aggregate costs into a comprehensive budget.

TRIP PROFILE: [insert full profile from Step 1]

YOUR TASKS:
1. Research typical costs at [destination] for [month]:
   - Accommodation (per night for [budget tier])
   - Food (meals per day: budget / mid-range / splurge)
   - Transport (daily transport costs, airport transfers)
   - Attractions (entrance fees for major sites)
   - Miscellaneous (SIM card, travel insurance, souvenirs)
2. Create itemized budget table by category
3. Show daily and total estimates for [duration]
4. Currency exchange tips (where to exchange, current rates)
5. Note where to save money and where not to skimp
6. Credit card / payment method advice (cash vs card acceptance)

OUTPUT FORMAT (in [language]):
- Budget Summary Table (daily + total)
- Detailed Breakdown by Category
- Money-Saving Tips
- Currency & Payment Guide
- Where to Splurge vs Save

CITE ALL SOURCES with URLs.
")
```

## Step 3: Source Triangulation

After all 5 agents return, cross-reference their findings:
- Flag contradictions between agents (e.g., different opening hours)
- Verify critical claims that appear in only one agent's output
- Merge overlapping recommendations
- Resolve any conflicts by preferring the most-cited or most-recent source

## Step 4: Synthesis

Combine all agent findings into a unified travel plan. Ensure:
- Every recommendation includes reasoning ("why this is recommended")
- Ratings/rankings from multiple sources are compared
- Clear Must-Do / Optional / Avoid categorization throughout
- The plan accounts for the specific group, interests, and constraints

## Step 5: Budget Consolidation

Create the final itemized budget table incorporating real costs from all agents:
- Accommodation total
- Food total (with daily breakdown)
- Transport total
- Attractions total
- Miscellaneous
- **Grand total** with per-person and per-day averages

## Step 6: Output Generation

Create the output directory and write all files:

```
TRAVEL/[destination]/
├── README.md                    # Overview & navigation to all files
├── itinerary.md                 # Day-by-day plan with reasoning
├── food_guide.md                # Complete food report
├── hotel_recommendations.md     # Hotel picks with budget analysis
├── transportation.md            # Getting around + parking
├── safety_and_tips.md           # Warnings, tips, cultural notes, packing
├── budget_summary.md            # Itemized budget table
├── quick_reference.md           # One-page cheat sheet
└── sources/
    └── bibliography.md          # All source URLs organized by topic
```

### File specifications:

**README.md**: Trip overview, quick stats (destination, dates, budget, group), table of contents linking to all other files.

**itinerary.md**: Day-by-day with Morning/Afternoon/Evening structure. Each activity has: description, why recommended, duration, cost, tips. Includes seasonal events, photography spots, day trips.

**food_guide.md**: Signature dishes with history, restaurant table (name, category, price, location, specialty, reservation needed), food markets, dining etiquette.

**hotel_recommendations.md**: 3-5 hotel options with: name, price/night, location, pros, cons, booking link. Ranked by value for the group type.

**transportation.md**: Getting there, getting around, parking guide (if driving), transport cards, apps, driving tips.

**safety_and_tips.md**: Safety warnings, scam alerts, weather + packing list, visa info, cultural etiquette, emergency numbers, SIM/WiFi, useful phrases, outlet/voltage info.

**budget_summary.md**: Itemized table with daily and total costs. Currency guide. Money-saving tips. Splurge vs save advice.

**quick_reference.md**: Single-page cheat sheet with: emergency numbers, key phrases, hotel address (for taxi), embassy address, WiFi info, key app names, currency quick-reference.

**sources/bibliography.md**: All URLs organized by topic (attractions, food, hotels, transport, safety).

## Step 7: Summary Output

After writing all files, print a concise summary to the user:

```
✅ Travel plan for [destination] is ready!

📁 Files written to: TRAVEL/[destination]/

📅 Itinerary Overview:
| Day | Highlights |
|-----|------------|
| Day 1 | ... |
| Day 2 | ... |
| ... | ... |

💰 Estimated Total Budget: [amount] [currency]
- Accommodation: [amount]
- Food: [amount]
- Transport: [amount]
- Attractions: [amount]
- Other: [amount]

📖 Read the full plan: TRAVEL/[destination]/README.md
```
