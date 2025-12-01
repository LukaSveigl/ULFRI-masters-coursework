<script lang="ts">
    import AppIcon from "$lib/components/AppIcon.svelte";
    import { formatMoney } from "$lib/formatting";
    import type { ClassValue } from "clsx";
    import { type Icon } from "lucide-svelte";
    import type { Component, ComponentProps } from "svelte";

    let {
        icon,
        n,
        typeString,
        value,
        class: klass
    }: {
        icon: Component<ComponentProps<Icon>> | typeof Icon;
        n: number;
        typeString: string;
        value: number;
        class?: ClassValue;
    } = $props();
</script>

<div class={["grid h-16 rounded-md font-semibold shadow-md", klass]}>
    {#if n > 0}
        <div class="flex flex-col items-start justify-between">
            <div class="flex items-center gap-2 pl-2 pt-2">
                <AppIcon {icon} strokeWidth="1.5" class="size-6" />
                <p class="text-sm leading-none">{n} {typeString} loan{n > 1 ? "s" : ""}</p>
            </div>
            <p class="pb-[10px] pl-2 text-lg leading-none">
                {formatMoney(value)}
            </p>
        </div>
    {:else}
        <div class="flex items-center gap-2 p-2">
            <AppIcon {icon} strokeWidth="1.5" class="size-10" />
            <p class="text-sm">0 {typeString} loans</p>
        </div>
    {/if}
</div>
