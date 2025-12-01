<script lang="ts">
    import type { OverrideSvelteHTMLElement } from "$lib/utils.svelte";
    import type { ClassValue } from "clsx";
    import type { Snippet } from "svelte";

    let {
        children,
        label,
        labelClass,
        otherWayAround,
        class: klass,
        ...props
    }: OverrideSvelteHTMLElement<
        "label",
        {
            children: Snippet<[]>;
            label: Snippet<[]> | string;
            labelClass?: ClassValue;
            otherWayAround?: boolean;
        }
    > = $props();
</script>

{#snippet span()}
    <span
        class={[
            "text-sm font-medium leading-none group-has-[:disabled]:cursor-not-allowed group-has-[:disabled]:opacity-70",
            labelClass
        ]}
    >
        {#if typeof label == "string"}
            {label}
        {:else}
            {@render label()}
        {/if}
    </span>
{/snippet}

<label {...props} class={["group", klass]}>
    {#if !otherWayAround}
        {@render span()}
        {@render children()}
    {:else}
        {@render children()}
        {@render span()}
    {/if}
</label>
