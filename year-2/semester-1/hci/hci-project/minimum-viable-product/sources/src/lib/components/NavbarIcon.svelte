<script lang="ts">
    import type { Snippet } from "svelte";

    let {
        href,
        active = false,
        children
    }: {
        href: string;
        active: boolean;
        children: Snippet;
    } = $props();
</script>

<a {href}>
    {@render children()}
</a>
