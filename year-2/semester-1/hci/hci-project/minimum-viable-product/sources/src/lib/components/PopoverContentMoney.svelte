<script lang="ts">
    import AppIcon from "$lib/components/AppIcon.svelte";
    import InputEur from "$lib/components/InputEur.svelte";
    import { Button } from "$lib/shadcn/components/ui/button";
    import { Check } from "lucide-svelte";

    let {
        startingValue,
        maxValue,
        onsubmit
    }: {
        startingValue: number;
        maxValue: number;
        onsubmit: (value: number) => void;
    } = $props();

    let value: number | null = $state(startingValue == 0 ? null : startingValue);
</script>

<form
    class="grid grid-cols-[1fr_auto] gap-4"
    onsubmit={(e) => {
        e.preventDefault();
        if (value != null) {
            onsubmit(value);
        }
    }}
>
    <InputEur max={maxValue} required bind:value />
    <Button class="flex-none shadow-md" variant="outline" size="icon" type="submit">
        <AppIcon icon={Check} class="!size-6" />
    </Button>
</form>
