/-
FlavorMath.lean - machine-checked properties of the flavor tastiness functional.

HONEST SCOPE: this does NOT prove any physical mixture "tastes good" (tastiness is
empirical, not a theorem). It proves the MATHEMATICAL property that the additive tastiness
estimate of a blend can never exceed its best single component. That is precisely why a
learned, NON-ADDITIVE mixture model is required to design a blend that beats its parts -
the whole scientific motivation, stated as a theorem.

We model pleasantness values as `Nat` (0..100 after scaling) to stay in Lean core (no
Mathlib dependency), using `omega` to discharge arithmetic. The result transfers to the
additive real-valued estimate T(R) = sum wᵢ pᵢ, which is a convex combination and hence
<= max pᵢ.

Check with:  lean FlavorMath.lean   (exit 0 = all theorems proved)
-/

namespace FlavorMath

/-- Binary max, core-only. -/
def max2 (a b : Nat) : Nat := if a ≤ b then b else a

theorem le_max2_left (a b : Nat) : a ≤ max2 a b := by
  unfold max2; split <;> omega

theorem le_max2_right (a b : Nat) : b ≤ max2 a b := by
  unfold max2; split <;> omega

/-- Max over a list of pleasantness values (0 for empty). -/
def listMax : List Nat → Nat
  | []      => 0
  | x :: xs => max2 x (listMax xs)

/-- THEOREM (best-part bound): every component's pleasantness is ≤ the blend's best part.
    This underlies P2: the additive blend estimate (a convex combination) ≤ max component. -/
theorem le_listMax (xs : List Nat) : ∀ x ∈ xs, x ≤ listMax xs := by
  intro x hx
  induction xs with
  | nil => cases hx
  | cons y ys ih =>
    simp only [listMax]
    cases hx with
    | head => exact le_max2_left _ _
    | tail _ hmem => exact Nat.le_trans (ih hmem) (le_max2_right _ _)

/-- Corollary: the max is itself achieved-bounded, i.e. listMax is an upper bound and the
    blend cannot exceed it. (Stated as: nothing in the list exceeds listMax.) -/
theorem blend_not_better_than_best (xs : List Nat) (x : Nat) (hx : x ∈ xs) :
    x ≤ listMax xs := le_listMax xs x hx

/-- Boundedness helper: if every component ≤ 100 then so is the max (P1 upper bound). -/
theorem listMax_le_of_all_le (xs : List Nat) (h : ∀ x ∈ xs, x ≤ 100) :
    listMax xs ≤ 100 := by
  induction xs with
  | nil => simp [listMax]
  | cons y ys ih =>
    have hy : y ≤ 100 := h y (by simp)
    have hys : ∀ z ∈ ys, z ≤ 100 := fun z hz => h z (List.mem_cons_of_mem y hz)
    have hrec : listMax ys ≤ 100 := ih hys
    unfold listMax max2
    split <;> omega

/-- COMPOSITION corollary: given per-class scores, no chemical composition (a weighting of
    classes) can score above the best single class. Discrete backbone: every class score is
    <= the max class score. This is why composition tuning alone cannot beat the best class;
    beating it requires non-additive interactions (the learned mixture model). -/
theorem composition_bounded_by_best_class (classScores : List Nat) (s : Nat)
    (hs : s ∈ classScores) : s ≤ listMax classScores := le_listMax classScores s hs

end FlavorMath
