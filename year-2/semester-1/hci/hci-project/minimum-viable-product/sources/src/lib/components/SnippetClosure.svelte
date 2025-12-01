<script module lang="ts">
    import { mkClosure } from "$lib/components/Closure.svelte";
    import SnippetClosure from "$lib/components/SnippetClosure.svelte";

    export function mkSnippetClosure<S extends Snippet<[any]>>(
        snippet: S,
        arg: S extends Snippet<[infer P]> ? P : never
    ) {
        return mkClosure(SnippetClosure, { snippet, arg });
    }
</script>

<script lang="ts">
    import type { Snippet } from "svelte";

    let { snippet, arg }: { snippet: Snippet<[any]>; arg: any } = $props();
</script>

{@render snippet(arg)}
