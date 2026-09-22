"""Self-contained 3v3 singles battle simulator with a damage-race heuristic teacher.

Contract (this file):
- Pure Python, standard library only. Every battle is fully determined by its
  integer seed through a single random.Random(seed) stream (team sampling,
  damage rolls, speed-tie coin flips); the epsilon-greedy exploration stream is
  derived from the same seed (see run_episode), so regeneration is exact.
- Roster: 17 real species with real base stats / types and 3-4 real damaging
  moves each, all at level 50 (IV 31, EV 0, neutral nature).
- Damage follows the main-series formula
  ((2*level/5+2)*power*A/D/50+2)*STAB*type_mult*roll with roll ~ U(0.85, 1.0).
  Speed decides move order (ties: battle RNG coin flip). Switching costs the
  turn: the foe lands one free hit on the incoming pokemon.
- Simplifications, stated honestly: no status moves, no priority, no recoil,
  no accuracy checks (every move hits), no items / abilities / weather; every
  move is a plain damaging move. This keeps the damage-race arithmetic the
  model sees in obs/candidates exactly the arithmetic the teacher uses.
- Teacher (our side), heuristic damage-race, in priority order:
  1) secure first-KO: if a move's minimum roll already KOs the foe active and
     we strike first (strictly faster, or the foe cannot KO us this turn),
     use the strongest such move;
  2) race rescue: if we are losing the damage race but some reserve mon --
     after paying the incoming free hit -- would win its own race against the
     foe active, switch to the best such reserve (margin = foe hits to die
     minus our hits to KO, +0.5 speed bonus, must be > 0);
  3) otherwise use the move with the highest expected damage (mean roll 0.925).
  Forced replacements after our faint are picked by the same race heuristic
  and are not recorded as model decisions.
- Foe policy: a fixed greedy AI that always uses its highest-expected-damage
  move and never switches voluntarily; replacements arrive in team order.
"""
import math
import random

LEVEL = 50
IV = 31
MAX_TURNS = 200
EXPECT_ROLL = 0.925  # mean of the uniform 0.85..1.0 damage roll
# Balance knob (task item 7): with fully real stats/moves at level 50, KOs took
# ~2 hits and mean battle length was ~8 turns. Scaling max HP by 2.5 slows the
# KO pace to ~4-5 hits per KO and brings mean battle length to ~17 turns.
# Base stats, level, move powers and the damage formula itself stay untouched.
HP_SCALE = 2.5

TYPE_CHART = {
    "Normal": {"Rock": 0.5, "Ghost": 0.0, "Steel": 0.5},
    "Fire": {"Fire": 0.5, "Water": 0.5, "Grass": 2.0, "Ice": 2.0, "Bug": 2.0,
             "Rock": 0.5, "Dragon": 0.5, "Steel": 2.0},
    "Water": {"Fire": 2.0, "Water": 0.5, "Grass": 0.5, "Ground": 2.0,
              "Rock": 2.0, "Dragon": 0.5},
    "Electric": {"Water": 2.0, "Electric": 0.5, "Grass": 0.5, "Ground": 0.0,
                 "Flying": 2.0, "Dragon": 0.5},
    "Grass": {"Fire": 0.5, "Water": 2.0, "Grass": 0.5, "Poison": 0.5,
              "Ground": 2.0, "Flying": 0.5, "Bug": 0.5, "Rock": 2.0,
              "Dragon": 0.5, "Steel": 0.5},
    "Ice": {"Fire": 0.5, "Water": 0.5, "Grass": 2.0, "Ice": 0.5, "Ground": 2.0,
            "Flying": 2.0, "Dragon": 2.0, "Steel": 0.5},
    "Fighting": {"Normal": 2.0, "Ice": 2.0, "Poison": 0.5, "Flying": 0.5,
                 "Psychic": 0.5, "Bug": 0.5, "Rock": 2.0, "Ghost": 0.0,
                 "Dark": 2.0, "Steel": 2.0, "Fairy": 0.5},
    "Poison": {"Grass": 2.0, "Poison": 0.5, "Ground": 0.5, "Rock": 0.5,
               "Ghost": 0.5, "Steel": 0.0, "Fairy": 2.0},
    "Ground": {"Fire": 2.0, "Electric": 2.0, "Grass": 0.5, "Poison": 2.0,
               "Flying": 0.0, "Bug": 0.5, "Rock": 2.0, "Steel": 2.0},
    "Flying": {"Electric": 0.5, "Grass": 2.0, "Fighting": 2.0, "Bug": 2.0,
               "Rock": 0.5, "Steel": 0.5},
    "Psychic": {"Fighting": 2.0, "Poison": 2.0, "Psychic": 0.5, "Dark": 0.0,
                "Steel": 0.5},
    "Bug": {"Fire": 0.5, "Grass": 2.0, "Fighting": 0.5, "Poison": 0.5,
            "Flying": 0.5, "Psychic": 2.0, "Ghost": 0.5, "Dark": 2.0,
            "Steel": 0.5, "Fairy": 0.5},
    "Rock": {"Fire": 2.0, "Ice": 2.0, "Fighting": 0.5, "Ground": 0.5,
             "Flying": 2.0, "Bug": 2.0, "Steel": 0.5},
    "Ghost": {"Normal": 0.0, "Psychic": 2.0, "Ghost": 2.0, "Dark": 0.5},
    "Dragon": {"Dragon": 2.0, "Steel": 0.5, "Fairy": 0.0},
    "Dark": {"Fighting": 0.5, "Psychic": 2.0, "Ghost": 2.0, "Dark": 0.5,
             "Fairy": 0.5},
    "Steel": {"Fire": 0.5, "Water": 0.5, "Electric": 0.5, "Ice": 2.0,
              "Rock": 2.0, "Steel": 0.5, "Fairy": 2.0},
    "Fairy": {"Fire": 0.5, "Fighting": 2.0, "Poison": 0.5, "Dragon": 2.0,
              "Dark": 2.0, "Steel": 0.5},
}

# species: (types, base stats (HP, Atk, Def, SpA, SpD, Spe), moves)
# move: (name, power, type, "phys" | "sp")
SPECIES = {
    "Snorlax": (("Normal",), (160, 110, 65, 65, 110, 30),
                (("Body Slam", 85, "Normal", "phys"),
                 ("Earthquake", 100, "Ground", "phys"),
                 ("Crunch", 80, "Dark", "phys"),
                 ("Fire Punch", 75, "Fire", "phys"))),
    "Blissey": (("Normal",), (255, 10, 10, 75, 135, 55),
                (("Psychic", 90, "Psychic", "sp"),
                 ("Ice Beam", 90, "Ice", "sp"),
                 ("Flamethrower", 90, "Fire", "sp"),
                 ("Shadow Ball", 80, "Ghost", "sp"))),
    "Lapras": (("Water", "Ice"), (130, 85, 80, 85, 95, 60),
               (("Surf", 90, "Water", "sp"),
                ("Ice Beam", 90, "Ice", "sp"),
                ("Thunderbolt", 95, "Electric", "sp"),
                ("Body Slam", 85, "Normal", "phys"))),
    "Milotic": (("Water",), (95, 60, 79, 100, 125, 81),
                (("Surf", 90, "Water", "sp"),
                 ("Ice Beam", 90, "Ice", "sp"),
                 ("Dragon Pulse", 85, "Dragon", "sp"))),
    "Gastrodon": (("Water", "Ground"), (111, 118, 93, 83, 105, 60),
                  (("Scald", 80, "Water", "sp"),
                   ("Earth Power", 90, "Ground", "sp"),
                   ("Ice Beam", 90, "Ice", "sp"),
                   ("Body Slam", 85, "Normal", "phys"))),
    "Magnezone": (("Electric", "Steel"), (70, 70, 115, 130, 90, 60),
                  (("Thunderbolt", 95, "Electric", "sp"),
                   ("Flash Cannon", 80, "Steel", "sp"),
                   ("Signal Beam", 80, "Bug", "sp"))),
    "Venusaur": (("Grass", "Poison"), (80, 82, 83, 100, 100, 100),
                 (("Energy Ball", 90, "Grass", "sp"),
                  ("Sludge Bomb", 90, "Poison", "sp"),
                  ("Earth Power", 90, "Ground", "sp"))),
    "Charizard": (("Fire", "Flying"), (78, 84, 78, 109, 85, 100),
                  (("Flamethrower", 90, "Fire", "sp"),
                   ("Air Slash", 75, "Flying", "sp"),
                   ("Dragon Pulse", 85, "Dragon", "sp"))),
    "Arcanine": (("Fire",), (90, 110, 80, 100, 80, 95),
                 (("Flamethrower", 90, "Fire", "sp"),
                  ("Crunch", 80, "Dark", "phys"),
                  ("Bulldoze", 60, "Ground", "phys"),
                  ("Iron Head", 80, "Steel", "phys"))),
    "Scizor": (("Bug", "Steel"), (70, 130, 100, 55, 80, 65),
               (("X-Scissor", 80, "Bug", "phys"),
                ("Iron Head", 80, "Steel", "phys"),
                ("Night Slash", 70, "Dark", "phys"),
                ("Aerial Ace", 60, "Flying", "phys"))),
    "Metagross": (("Steel", "Psychic"), (80, 135, 130, 95, 90, 70),
                  (("Meteor Mash", 85, "Steel", "phys"),
                   ("Earthquake", 100, "Ground", "phys"),
                   ("Zen Headbutt", 80, "Psychic", "phys"),
                   ("Brick Break", 75, "Fighting", "phys"))),
    "Tyranitar": (("Rock", "Dark"), (100, 134, 110, 95, 100, 61),
                  (("Crunch", 80, "Dark", "phys"),
                   ("Earthquake", 100, "Ground", "phys"),
                   ("Rock Slide", 75, "Rock", "phys"),
                   ("Iron Head", 80, "Steel", "phys"))),
    "Garchomp": (("Dragon", "Ground"), (108, 130, 95, 80, 85, 102),
                 (("Dragon Claw", 80, "Dragon", "phys"),
                  ("Earthquake", 100, "Ground", "phys"),
                  ("Iron Head", 80, "Steel", "phys"),
                  ("Fire Fang", 65, "Fire", "phys"))),
    "Gardevoir": (("Psychic", "Fairy"), (68, 65, 65, 125, 115, 80),
                  (("Psychic", 90, "Psychic", "sp"),
                   ("Moonblast", 95, "Fairy", "sp"),
                   ("Shadow Ball", 80, "Ghost", "sp"),
                   ("Thunderbolt", 95, "Electric", "sp"))),
    "Togekiss": (("Fairy", "Flying"), (85, 50, 95, 120, 115, 80),
                 (("Air Slash", 75, "Flying", "sp"),
                  ("Dazzling Gleam", 80, "Fairy", "sp"),
                  ("Flamethrower", 90, "Fire", "sp"),
                  ("Aura Sphere", 80, "Fighting", "sp"))),
    "Jolteon": (("Electric",), (65, 65, 60, 110, 95, 130),
                (("Thunderbolt", 95, "Electric", "sp"),
                 ("Shadow Ball", 80, "Ghost", "sp"),
                 ("Swift", 60, "Normal", "sp"))),
    "Gallade": (("Psychic", "Fighting"), (68, 125, 65, 65, 115, 80),
                (("Psycho Cut", 70, "Psychic", "phys"),
                 ("Leaf Blade", 90, "Grass", "phys"),
                 ("Night Slash", 70, "Dark", "phys"),
                 ("Close Combat", 120, "Fighting", "phys"))),
}


def type_mult(move_type, def_types):
    """Product of the chart entries for each defending type (default 1x)."""
    m = 1.0
    for t in def_types:
        m *= TYPE_CHART[move_type].get(t, 1.0)
    return m


def stat_hp(base, level=LEVEL):
    return (2 * base + IV) * level // 100 + level + 10


def stat_other(base, level=LEVEL):
    return (2 * base + IV) * level // 100 + 5


def hits_to_ko(defender, frac):
    """Hits needed at expected per-hit fraction `frac` (of max HP) to KO from current HP."""
    if frac <= 0.0:
        return 99
    return min(99, max(1, math.ceil(defender.hp / (frac * defender.max_hp))))


class Mon:
    __slots__ = ("species", "types", "moves", "max_hp", "hp", "atk", "dfn",
                 "spa", "spd", "spe")

    def __init__(self, species, level=LEVEL):
        types, base, moves = SPECIES[species]
        self.species, self.types, self.moves = species, types, moves
        self.max_hp = int(stat_hp(base[0], level) * HP_SCALE)
        self.hp = self.max_hp
        self.atk = stat_other(base[1], level)
        self.dfn = stat_other(base[2], level)
        self.spa = stat_other(base[3], level)
        self.spd = stat_other(base[4], level)
        self.spe = stat_other(base[5], level)

    @property
    def alive(self):
        return self.hp > 0

    def pct(self):
        return int(round(100.0 * self.hp / self.max_hp))


def _pct_of(dmg, target):
    return int(round(100.0 * dmg / target.max_hp))


def _hits_txt(n):
    return f"{n} hit{'s' if n != 1 else ''}"


class Battle:
    """One 3v3 singles battle; step() takes exactly one player decision."""

    def __init__(self, our_species, foe_species, rng, level=LEVEL, max_turns=MAX_TURNS):
        self.our = [Mon(s, level) for s in our_species]
        self.foe = [Mon(s, level) for s in foe_species]
        self.rng = rng
        self.level = level
        self.max_turns = max_turns
        self.turn = 0
        self.our_idx = 0
        self.foe_idx = 0
        self.done = False
        self.success = False

    # ---------------- rules ----------------
    def _stab(self, attacker, move):
        return 1.5 if move[2] in attacker.types else 1.0

    @staticmethod
    def _atk_of(attacker, move):
        return attacker.spa if move[3] == "sp" else attacker.atk

    @staticmethod
    def _def_of(defender, move):
        return defender.spd if move[3] == "sp" else defender.dfn

    def expected_frac(self, attacker, defender, move):
        """Expected damage as a fraction of the defender's MAX hp (point estimate)."""
        mult = type_mult(move[2], defender.types)
        if mult == 0.0:
            return 0.0
        a, d = self._atk_of(attacker, move), self._def_of(defender, move)
        raw = ((22.0 * move[1] * a / d) / 50.0 + 2.0) * self._stab(attacker, move) * mult * EXPECT_ROLL
        return raw / defender.max_hp

    def _best(self, attacker, defender):
        """(index, move, expected_frac) of the highest-expected-damage move; first on ties."""
        best = (-1, None, -1.0)
        for i, mv in enumerate(attacker.moves):
            f = self.expected_frac(attacker, defender, mv)
            if f > best[2]:
                best = (i, mv, f)
        return best

    def roll_damage(self, attacker, defender, move):
        mult = type_mult(move[2], defender.types)
        if mult == 0.0:
            return 0
        a, d = self._atk_of(attacker, move), self._def_of(defender, move)
        base = (22 * move[1] * a // d) // 50 + 2
        dmg = max(1, int(base * self._stab(attacker, move) * mult))
        return max(1, int(dmg * self.rng.uniform(0.85, 1.0)))

    # ---------------- text surfaces ----------------
    def observation(self):
        me, op = self.our[self.our_idx], self.foe[self.foe_idx]
        rel = ("we are faster" if me.spe > op.spe
               else "they are faster" if op.spe > me.spe else "speeds are tied")
        ests = "; ".join(
            f"{i + 1}) {mv[0]}: about {int(round(self.expected_frac(me, op, mv) * 100))} percent of its HP"
            for i, mv in enumerate(me.moves))
        oi, omv, ofrac = self._best(me, op)
        fi, fmv, ffrac = self._best(op, me)
        hw, ht = hits_to_ko(op, ofrac), hits_to_ko(me, ffrac)
        bench = [m for j, m in enumerate(self.our) if j != self.our_idx and m.alive]
        bench_txt = ", ".join(
            f"{m.species} ({'/'.join(m.types)}) at {m.pct()} percent HP" for m in bench) or "none"
        foes_left = sum(1 for m in self.foe if m.alive) - 1
        return (f"Pokemon 3v3 singles at level {self.level}, turn {self.turn + 1}. "
                f"Our active: {me.species} ({'/'.join(me.types)}), {me.pct()} percent HP, "
                f"{me.spe} speed; {rel}.\n"
                f"Foe active: {op.species} ({'/'.join(op.types)}), {op.pct()} percent HP, "
                f"{op.spe} speed.\n"
                f"Our moves vs {op.species}: {ests}.\n"
                f"Damage race with best moves ({omv[0]} vs {fmv[0]}): we need about "
                f"{_hits_txt(hw)} to KO {op.species}, they need about {_hits_txt(ht)} to KO "
                f"{me.species}.\n"
                f"Our bench: {bench_txt}. Foe reserve pokemon left: {foes_left}.")

    def candidates(self):
        me, op = self.our[self.our_idx], self.foe[self.foe_idx]
        cands = {}
        for i, mv in enumerate(me.moves):
            frac = self.expected_frac(me, op, mv)
            hits = hits_to_ko(op, frac)
            cands[f"move:{i + 1}"] = (
                f"Use {mv[0]} ({mv[2]}, power {mv[1]}, {type_mult(mv[2], op.types):g}x): about "
                f"{int(round(frac * 100))} percent of {op.species}'s HP; about {_hits_txt(hits)} to KO.")
        for idx, m in enumerate(self.our):
            if idx == self.our_idx or not m.alive:
                continue
            bi, bm, bf = self._best(m, op)
            fi, fmv, ff = self._best(op, m)
            free = int(round(ff * 100))
            hb = hits_to_ko(op, bf)
            hp_after = max(m.hp - ff * m.max_hp, 0.5)
            htb = min(99, max(1, math.ceil(hp_after / (ff * m.max_hp))))
            cands[f"switch:{idx}"] = (
                f"Switch to {m.species} ({'/'.join(m.types)}) at {m.pct()} percent HP: foe hits it "
                f"for free (about {free} percent); then {m.species} needs about {_hits_txt(hb)} to KO "
                f"{op.species} and is KOed in about {_hits_txt(htb)} from the foe.")
        return cands

    # ---------------- teacher ----------------
    def teacher_action(self):
        """Heuristic damage-race teacher; returns a key of self.candidates()."""
        me, op = self.our[self.our_idx], self.foe[self.foe_idx]
        we_faster = me.spe > op.spe
        fi, fmv, ffrac = self._best(op, me)
        foe_max = ffrac / EXPECT_ROLL * me.max_hp  # max-roll estimate
        foe_can_ko = foe_max >= me.hp
        # rule 1: a move whose minimum roll KOs the foe active, and we strike first.
        key, frac = None, -1.0
        for i, mv in enumerate(me.moves):
            f = self.expected_frac(me, op, mv)
            min_roll = f / EXPECT_ROLL * 0.85 * op.max_hp
            if min_roll >= op.hp and (we_faster or not foe_can_ko) and f > frac:
                key, frac = f"move:{i + 1}", f
        if key is not None:
            return key
        # rule 2: losing the race -> switch to the reserve with the best
        # post-free-hit margin (must be positive; switching while winning the
        # race measurably lowered the teacher's win rate, so it is ruled out).
        oi, omv, ofrac = self._best(me, op)
        hw, ht = hits_to_ko(op, ofrac), hits_to_ko(me, ffrac)
        winning = hw < ht or (hw == ht and we_faster)
        if not winning:
            best_idx, best_margin = None, 0.0
            for idx, m in enumerate(self.our):
                if idx == self.our_idx or not m.alive:
                    continue
                bi, bm, bf = self._best(m, op)
                gi, gmv, gf = self._best(op, m)
                if m.hp - gf * m.max_hp <= 0:
                    continue  # the free hit itself would KO the reserve
                hb = hits_to_ko(op, bf)
                htb = math.ceil((m.hp - gf * m.max_hp) / (gf * m.max_hp))
                margin = htb - hb + (0.5 if m.spe > op.spe else 0.0)
                if margin > best_margin:
                    best_idx, best_margin = idx, margin
            if best_idx is not None:
                return f"switch:{best_idx}"
        # rule 3: highest expected damage.
        return f"move:{oi + 1}"

    def _forced_replacement(self):
        """Bench slot our teacher sends in after a faint (race heuristic, no free hit)."""
        op = self.foe[self.foe_idx]
        best_idx, best_score = None, -1e9
        for idx, m in enumerate(self.our):
            if idx == self.our_idx or not m.alive:
                continue
            bi, bm, bf = self._best(m, op)
            gi, gmv, gf = self._best(op, m)
            score = hits_to_ko(m, gf) - hits_to_ko(op, bf) + (0.5 if m.spe > op.spe else 0.0)
            if score > best_score:
                best_idx, best_score = idx, score
        return best_idx

    # ---------------- resolution ----------------
    def _foe_send_next(self, parts):
        for j, m in enumerate(self.foe):
            if m.alive:
                self.foe_idx = j
                parts.append(f"Foe sent out {m.species}.")
                return
        self.done, self.success = True, True

    def _our_send_replacement(self, parts):
        idx = self._forced_replacement()
        if idx is None:
            self.done, self.success = True, False
        else:
            self.our_idx = idx
            parts.append(f"We sent out {self.our[idx].species}.")

    def _foe_attack(self, parts):
        """Foe active attacks our active; handles our faint + replacement. True if we fainted."""
        op, me = self.foe[self.foe_idx], self.our[self.our_idx]
        fi, fmv, _ = self._best(op, me)
        dmg = self.roll_damage(op, me, fmv)
        me.hp -= dmg
        if me.hp <= 0:
            parts.append(f"Foe {op.species} used {fmv[0]}: our {me.species} fainted.")
            self._our_send_replacement(parts)
            return True
        parts.append(f"Foe {op.species} used {fmv[0]}: dealt {_pct_of(dmg, me)} percent; "
                     f"our {me.species} at {me.pct()} percent.")
        return False

    def _resolve_our_move(self, move, parts):
        me, op = self.our[self.our_idx], self.foe[self.foe_idx]
        dmg = self.roll_damage(me, op, move)
        op.hp -= dmg
        if op.hp <= 0:
            parts.append(f"We used {move[0]}: {op.species} fainted.")
            self._foe_send_next(parts)
        else:
            parts.append(f"We used {move[0]}: dealt {_pct_of(dmg, op)} percent; "
                         f"{op.species} at {op.pct()} percent.")

    def step(self, action):
        if self.done:
            raise RuntimeError("step after terminal state")
        self.turn += 1
        parts = []
        if action.startswith("switch:"):
            idx = int(action.split(":")[1])
            if idx == self.our_idx or not self.our[idx].alive:
                raise ValueError(f"illegal switch target {action}")
            incoming = self.our[idx]
            self.our_idx = idx
            op = self.foe[self.foe_idx]
            fi, fmv, _ = self._best(op, incoming)
            dmg = self.roll_damage(op, incoming, fmv)
            incoming.hp -= dmg
            if incoming.hp <= 0:
                parts.append(f"We switched in {incoming.species}; foe {op.species} used "
                             f"{fmv[0]}: {incoming.species} fainted.")
                self._our_send_replacement(parts)
            else:
                parts.append(f"Switched in {incoming.species}; took a free hit for "
                             f"{_pct_of(dmg, incoming)} percent; {incoming.species} at "
                             f"{incoming.pct()} percent.")
        elif action.startswith("move:"):
            mi = int(action.split(":")[1]) - 1
            me, op = self.our[self.our_idx], self.foe[self.foe_idx]
            move = me.moves[mi]
            we_first = me.spe > op.spe or (me.spe == op.spe and self.rng.random() < 0.5)
            if we_first:
                self._resolve_our_move(move, parts)
                if not self.done and op.alive:
                    self._foe_attack(parts)
            else:
                fainted = self._foe_attack(parts)
                if not fainted and not self.done:
                    self._resolve_our_move(move, parts)
        else:
            raise ValueError(f"unknown action {action}")
        if not self.done and self.turn >= self.max_turns:
            self.done, self.success = True, False
            parts.append("Turn limit reached; counted as a loss.")
        if self.done:
            parts.append("We win the battle." if self.success else "We lose the battle.")
        return " ".join(parts)


def run_episode(seed, epsilon=0.0, level=LEVEL, max_turns=MAX_TURNS):
    """Roll one battle with the epsilon-greedy damage-race teacher; record everything.

    Team sampling and all battle randomness come from random.Random(seed); the
    exploration stream comes from random.Random((seed << 3) ^ int(epsilon*100)),
    mirroring gen_data.py, so a battle is a pure function of (seed, epsilon).
    """
    rng = random.Random(seed)
    roster = sorted(SPECIES)
    our_team = rng.sample(roster, 3)
    foe_team = rng.sample(roster, 3)
    env = Battle(our_team, foe_team, rng=rng, level=level, max_turns=max_turns)
    eps_rng = random.Random((seed << 3) ^ int(round(epsilon * 100)))
    steps = []
    while not env.done:
        obs, cands = env.observation(), env.candidates()
        action = env.teacher_action()
        if eps_rng.random() < epsilon:
            action = eps_rng.choice(list(cands))
        event = env.step(action)
        steps.append({"t": len(steps) + 1, "obs": obs, "candidates": cands,
                      "action": action, "event": event})
    return {"task": "pokemon", "episode_id": None, "split": None,
            "meta": {"seed": seed, "epsilon": epsilon, "level": level,
                     "max_turns": max_turns, "our_team": our_team, "foe_team": foe_team,
                     "teacher": "heuristic-damage-race",
                     "foe_policy": "greedy-max-damage-no-switch"},
            "success": env.success, "n_steps": len(steps), "steps": steps}
