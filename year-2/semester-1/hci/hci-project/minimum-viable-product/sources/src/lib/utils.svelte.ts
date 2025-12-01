import { untrack, type Component, type ComponentProps, type SvelteComponent } from "svelte";
import type { SvelteHTMLElements } from "svelte/elements";

export type Override<T, U, O extends string | number | symbol | never = never> = Omit<
    T,
    keyof U | O
> &
    U;
export type OverrideComponentProps<
    Comp extends SvelteComponent | Component<any, any>,
    U,
    O extends string | number | symbol | never = never
> = Override<ComponentProps<Comp>, U, O>;
export type OverrideSvelteHTMLElement<
    T extends keyof SvelteHTMLElements,
    U,
    O extends string | number | symbol | never = never
> = Override<SvelteHTMLElements[T], U, O>;

export function mutation<T, const StallReason>(
    config:
        | ((value: T) => Promise<StallReason | undefined>)
        | {
              get?: () => T;
              set: (value: T) => Promise<StallReason | undefined>;
              abort?: () => void;
          }
) {
    const cfg = "set" in config ? config : { set: config };

    let applyRequested = $state(false);
    let applyInput: T = $state()!;
    let applyInputSet = $state(false);
    let applyStall: StallReason | undefined = $state(undefined);
    let applyError: unknown = $state(undefined);
    let applyPromise: Promise<void> | null = $state(null);

    async function apply() {
        while (applyRequested) {
            applyRequested = false;
            try {
                applyStall = await cfg.set(applyInput);
            } catch (e) {
                applyError = e;
            }
        }
    }

    if (cfg.abort) {
        const abort = cfg.abort;
        $effect.pre(() => abort);
    }

    return {
        get settled() {
            return (
                !applyRequested &&
                !applyPromise &&
                applyStall === undefined &&
                applyError === undefined
            );
        },
        get stalled() {
            if (applyStall !== undefined) return "reason";
            if (applyError !== undefined) return "error";
            return false;
        },
        get stallReason(): StallReason {
            if (applyStall === undefined) throw new Error("not stalled with a reason");
            return applyStall;
        },
        get stallError(): unknown {
            if (applyError === undefined) throw new Error("not stalled with an error");
            return applyError;
        },
        get applying() {
            return !this.settled && !this.stalled;
        },
        get desiredValue() {
            if (!applyInputSet) throw new Error("desired value not set");
            return applyInput;
        },
        set desiredValue(value: T) {
            untrack(() => {
                applyRequested = true;
                applyInput = value;
                applyInputSet = true;
                applyStall = undefined;
                applyError = undefined;
                if (applyPromise) {
                    cfg.abort?.();
                } else {
                    applyPromise = apply();
                    applyPromise.finally(() => {
                        applyPromise = null;
                    });
                }
            });
        },
        get value() {
            if (!cfg.get) {
                throw new Error("value not available without get function");
            }
            return this.settled ? cfg.get() : this.desiredValue;
        },
        set value(value: T) {
            if (!cfg.get) {
                throw new Error("value not available without get function");
            }
            this.desiredValue = value;
        }
    };
}

// Because svelte doesn't let you do {@let ...}
export function sneakyState<T>(initial: T) {
    let value = $state(initial);
    return {
        get value() {
            return value;
        },
        set value(newValue) {
            value = newValue;
        }
    };
}
