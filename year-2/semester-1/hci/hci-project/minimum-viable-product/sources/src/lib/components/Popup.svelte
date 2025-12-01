<script module lang="ts">
    import { createContext } from "$lib/context";

    const nestingInfoCtx = createContext<{
        readonly level: number;
        readonly open: boolean;
    }>("nestingInfo");
</script>

<script lang="ts">
    import * as AlertDialog from "$lib/shadcn/components/ui/alert-dialog";
    import * as Dialog from "$lib/shadcn/components/ui/dialog";
    import type { OverrideComponentProps } from "$lib/utils.svelte";
    import type { Snippet } from "svelte";

    type CommonOverride = { children: Snippet<[]>; trigger?: Snippet<[]> };

    let {
        open = $bindable(false),
        alert,
        children,
        trigger,
        ...props
    }:
        | OverrideComponentProps<typeof Dialog.Root, { alert?: false } & CommonOverride>
        | OverrideComponentProps<
              typeof AlertDialog.Root,
              { alert: true } & CommonOverride
          > = $props();

    const parentNestingInfo = nestingInfoCtx.has() ? nestingInfoCtx.get() : null;
    const nestingLevel = (parentNestingInfo?.level ?? 0) + 1;
    let reallyOpen = {
        get value() {
            return open && (parentNestingInfo?.open ?? true);
        },
        set value(value) {
            open = value;
        }
    };
    nestingInfoCtx.set({
        level: nestingLevel,
        get open() {
            return reallyOpen.value;
        }
    });

    let nestingCls =
        {
            1: "max-w-80",
            2: "max-w-72",
            3: "max-w-64",
            4: "max-w-60"
        }[nestingLevel] ?? "max-w-80";
</script>

{#if !alert}
    <Dialog.Root bind:open={reallyOpen.value} {...props}>
        {@render trigger?.()}
        <Dialog.Content class={["rounded-md", nestingCls]}>
            {@render children()}
        </Dialog.Content>
    </Dialog.Root>
{:else}
    <AlertDialog.Root bind:open={reallyOpen.value} {...props}>
        {@render trigger?.()}
        <AlertDialog.Content
            class={["rounded-md", nestingCls]}
            escapeKeydownBehavior="close"
            interactOutsideBehavior="close"
        >
            {@render children()}
        </AlertDialog.Content>
    </AlertDialog.Root>
{/if}
