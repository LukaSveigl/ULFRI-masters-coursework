import { getContext, hasContext, setContext } from "svelte";

export function createContext<T>(name?: string | undefined | null): {
    get(): T;
    set(v: T): void;
    has(): boolean;
} {
    const sym = Symbol(name ?? undefined);
    return {
        get() {
            if (!hasContext(sym)) {
                throw TypeError(`tried to get non-existent context ${sym.toString()}`);
            }
            return getContext<T>(sym);
        },
        set(v) {
            setContext(sym, v);
        },
        has() {
            return hasContext(sym);
        }
    };
}
