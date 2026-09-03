# Travel Planner - Multi-Agent Travel Planning System

A comprehensive travel planning skill for Claude Code that coordinates multiple specialist research agents to produce well-sourced, detailed travel plans with itineraries, food guides, logistics, safety tips, and budgets.

## Overview

Travel Planner is an orchestration skill that deploys 5 specialized research agents simultaneously to gather comprehensive travel information. Each agent focuses on a specific domain (attractions, food, logistics, safety, budget), and their findings are synthesized into a unified, actionable travel plan.

**Key Features:**

- 🎯 **Parallel Research** - 5 specialist agents work simultaneously for faster results
- 📍 **Comprehensive Coverage** - Attractions, food, transport, safety, and budget in one plan
- 🔍 **Source Triangulation** - Cross-reference findings across multiple web sources
- 📊 **Structured Output** - Organized markdown files ready for reference
- 💡 **Reasoning-Based** - Every recommendation includes explanation and context
- 🌍 **Web-Powered** - Uses WebSearch and WebFetch for up-to-date information

## Architecture

### Multi-Agent System

```
┌─────────────────────────────────────────────────────────────┐
│                    Travel Planner (Orchestrator)            │
└────────────┬───────────────────────────────────────┬────────┘
             │                                       │
    ┌────────▼────────┐                   ┌─────────▼─────────┐
    │ trip-architect  │                   │  food-explorer     │
    │ Itineraries &   │                   │  Local cuisine &   │
    │ attractions     │                   │  restaurants       │
    └─────────────────┘                   └────────────────────┘
             │                                       │
    ┌────────▼────────┐                   ┌─────────▼─────────┐
    │logistics-planner│                   │   local-intel      │
    │ Transport &     │                   │  Safety & tips     │
    │ hotels          │                   │  culture           │
    └─────────────────┘                   └────────────────────┘
             │                                       │
             └───────────────┬───────────────────────┘
                             │
                    ┌────────▼────────┐
                    │budget-calculator│
                    │  Cost breakdown │
                    └─────────────────┘
```

### Specialist Agents

| Agent | Role | Research Areas |
|-------|------|----------------|
| **trip-architect** | Itinerary Design | Attractions, opening hours, optimal visit order, seasonal relevance, route planning |
| **food-explorer** | Culinary Research | Signature dishes, restaurants, food markets, street food, dining culture |
| **logistics-planner** | Transportation | Transit options, parking, hotel recommendations, practical logistics |
| **local-intel** | Safety & Culture | Safety warnings, scams, weather, visa requirements, cultural etiquette, emergency info |
| **budget-calculator** | Cost Aggregation | Budget breakdown, daily estimates, currency tips, money-saving advice |

## Installation

### Prerequisites

- [Claude Code CLI](https://github.com/anthropics/claude-code) installed
- Skills directory configured (`~/.claude/skills/` on macOS/Linux)

### Install Steps

1. **Clone the repository:**
   ```bash
   cd ~/.claude/skills/
   git clone https://github.com/ErlebnisW/travel-planner.git
   ```

2. **Verify installation:**
   ```bash
   ls ~/.claude/skills/travel-planner/
   # Should show: SKILL.md, skills/, commands/, settings.local.json
   ```

3. **Restart Claude Code** to load the new skill

### Manual Installation

Alternatively, download and extract to `~/.claude/skills/travel-planner/`:

```bash
mkdir -p ~/.claude/skills/travel-planner
cd ~/.claude/skills/travel-planner
# Copy all files from this repository
```

## Usage

### Basic Usage

Simply tell Claude you want to plan a trip:

```
Plan a 5-day trip to Kyoto in April
```

```
I'm traveling to Paris for 3 days in October, I love food and art museums
```

```
Help me plan a weekend trip to San Diego with my family
```

### Input Format

The skill accepts free-form natural language descriptions. It automatically extracts:

- **Destination** - City or region
- **Duration** - Number of days
- **Timing** - Month or season
- **Preferences** - Interests mentioned (food, hiking, museums, etc.)

### Example Session

```
You: Plan a 4-day trip to Rome in June

Claude: I'm activating the travel-planner skill to create a comprehensive
travel plan for Rome. I'll deploy 5 specialist agents to research:
- Attractions and day-by-day itinerary
- Local cuisine and restaurant recommendations
- Transportation and hotel options
- Safety tips and cultural etiquette
- Complete budget breakdown

[Agents work in parallel...]

Travel plan complete! I've created structured files in TRAVEL/rome/:
✓ itinerary.md - 4-day route with attractions
✓ food_guide.md - Roman specialties and restaurants
✓ hotel_recommendations.md - 3 options within budget
✓ transportation.md - Metro, walking routes, airport transfer
✓ safety_and_tips.md - Scam warnings, dress codes, emergency contacts
✓ budget_summary.md - €800-1200 total estimate
✓ quick_reference.md - At-a-glance essentials
```

## Output Structure

All plans are saved to `TRAVEL/[destination]/` with the following files:

```
TRAVEL/rome/
├── README.md                    # Overview and getting started
├── itinerary.md                 # Day-by-day schedule with attractions
├── food_guide.md                # Local dishes, restaurants, food culture
├── hotel_recommendations.md     # 3 hotel options with pros/cons
├── transportation.md            # Metro, buses, parking, airport transfer
├── safety_and_tips.md          # Safety, scams, weather, cultural notes
├── budget_summary.md           # Itemized costs and money-saving tips
├── quick_reference.md          # Essential info at a glance
└── sources/
    └── bibliography.md         # All web sources used
```

### File Descriptions

| File | Contents |
|------|----------|
| **README.md** | Trip overview, best time to visit, highlights, preparation checklist |
| **itinerary.md** | Day-by-day schedule with attraction details, opening hours, visit duration, reasoning for order |
| **food_guide.md** | Signature dishes, restaurant recommendations (budget/mid/upscale), food markets, dining etiquette |
| **hotel_recommendations.md** | 3 hotel options with location analysis, price range, booking tips |
| **transportation.md** | Transit cards, metro routes, parking info, airport transfers, inter-city travel |
| **safety_and_tips.md** | Safety warnings, common scams, emergency contacts, visa info, cultural do's/don'ts, packing list |
| **budget_summary.md** | Daily cost breakdown, total estimate ranges, currency exchange tips, money-saving strategies |
| **quick_reference.md** | Emergency numbers, key phrases, essential apps, quick facts |
| **sources/bibliography.md** | All web sources with URLs and access dates for fact-checking |

## Configuration

### Permissions (settings.local.json)

The skill requires web access for research:

```json
{
  "permissions": {
    "allow": [
      "Bash(ls:*)",
      "Bash(done:*)",
      "WebSearch",
      "WebFetch",
      "mcp__exa__web_search_exa"
    ]
  }
}
```

These permissions allow agents to:
- Search the web for travel information
- Fetch content from specific URLs
- Access the Exa AI search API (if configured)
- Execute basic bash commands for file operations

### Customization

To modify agent behavior, edit the individual skill files:

```
skills/
├── trip-architect/SKILL.md      # Customize itinerary preferences
├── food-explorer/SKILL.md       # Adjust food research focus
├── logistics-planner/SKILL.md   # Modify transport priorities
├── local-intel/SKILL.md         # Change safety research depth
└── budget-calculator/SKILL.md   # Adjust budget categories
```

## Advanced Usage

### Specialized Requests

The system handles specific preferences automatically:

```
Plan a 7-day Japan trip focused on temples and traditional culture
```

```
3-day Paris itinerary for food lovers on a budget
```

```
Family-friendly 5-day London trip with kids aged 8 and 10
```

### Multi-City Trips

For complex itineraries, break into separate requests:

```
First: Plan 3 days in Barcelona
Then: Plan 2 days in Madrid
```

### Research-Only Requests

You can invoke individual agents for specific research:

```
Research local food specialties in Tokyo
```

```
Find hotel options in downtown Seattle under $150/night
```

## How It Works

### 1. Input Parsing

The orchestrator extracts structured data from natural language:

```
"5-day trip to Kyoto in April" →
{
  destination: "Kyoto",
  duration: 5,
  month: "April",
  preferences: []
}
```

### 2. Parallel Agent Deployment

Five agents are spawned simultaneously using Claude Code's Task tool:

```typescript
Task(subagent_type="trip-architect", prompt="Research Kyoto attractions...")
Task(subagent_type="food-explorer", prompt="Research Kyoto cuisine...")
// ... 3 more agents
```

### 3. Source Triangulation

Each agent cross-references multiple sources:
- Official tourism websites
- Recent blog posts and travel guides
- Review platforms (TripAdvisor, Google Maps)
- Local government resources

### 4. Synthesis

The orchestrator combines findings, resolving conflicts and filling gaps:
- Cross-check attraction hours across sources
- Validate restaurant recommendations
- Ensure budget consistency
- Verify safety information accuracy

### 5. Structured Output

Results are formatted into markdown files with consistent structure, making them easy to reference during travel.

## Requirements

### System Requirements

- **Operating System:** macOS, Linux, or Windows with WSL
- **Claude Code:** Latest version recommended
- **Internet:** Stable connection for web research
- **Disk Space:** ~50MB per travel plan

### API Requirements

- Claude API access (via Claude Code)
- Optional: Exa AI API key for enhanced web search (set in MCP config)

## Troubleshooting

### Skill Not Loading

```bash
# Check skill is in correct location
ls ~/.claude/skills/travel-planner/SKILL.md

# Check sub-skills are present
ls ~/.claude/skills/travel-planner/skills/
```

### Permission Errors

Verify `settings.local.json` includes web permissions:

```bash
cat ~/.claude/skills/travel-planner/settings.local.json
```

### Incomplete Plans

If agents fail to complete research:
- Check internet connection
- Verify Claude Code has web access enabled
- Try a more specific destination ("Kyoto" vs "Japan")

## Contributing

Contributions are welcome! Areas for improvement:

- **Additional Agents:** Weather specialist, activity booker, photo spot finder
- **Enhanced Synthesis:** Better conflict resolution between sources
- **Template Customization:** Different output formats (PDF, JSON)
- **Multi-Language:** Support for non-English destinations
- **Offline Mode:** Cache common destination data

### Development Setup

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-agent`
3. Make changes to skills or add new agents
4. Test thoroughly with various destinations
5. Submit a pull request

### Adding a New Agent

1. Create new directory: `skills/your-agent-name/`
2. Add `SKILL.md` with agent role and research areas
3. Update main `SKILL.md` to include new agent in workflow
4. Add to permissions if new tools needed
5. Test integration with orchestrator

## License

MIT License - See LICENSE file for details

## Acknowledgments

- Built for Claude Code by Anthropic
- Uses web search capabilities for up-to-date travel information
- Inspired by multi-agent research workflows

## Support

- **Issues:** [GitHub Issues](https://github.com/ErlebnisW/travel-planner/issues)
- **Discussions:** [GitHub Discussions](https://github.com/ErlebnisW/travel-planner/discussions)
- **Claude Code Help:** [Claude Code Documentation](https://github.com/anthropics/claude-code)

---

**Ready to plan your next adventure?** Install the skill and tell Claude where you want to go! ✈️
