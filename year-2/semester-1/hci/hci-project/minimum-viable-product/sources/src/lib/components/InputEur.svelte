<script lang="ts">
    import { Input } from "$lib/shadcn/components/ui/input";
    import { type OverrideComponentProps } from "$lib/utils.svelte";

    let {
        value = $bindable(),
        class: klass,
        ...props
    }: OverrideComponentProps<typeof Input, { value?: number | null }, "type" | "step"> = $props();
</script>

<div class={["grid grid-cols-[1fr_auto] items-stretch", klass]}>
    <Input
        min="0.00"
        {...props}
        class="peer z-10 rounded-r-none border-r-0"
        type="number"
        step="0.01"
        bind:value
    />
    <div
        class="flex aspect-square size-10 items-center justify-center rounded-r-md border border-input peer-disabled:cursor-not-allowed peer-disabled:opacity-50"
    >
        <p class="text-lg font-medium leading-none">€</p>
    </div>
</div>
