<script lang="ts">
    import Popup from "$lib/components/Popup.svelte";
    import * as AlertDialog from "$lib/shadcn/components/ui/alert-dialog";
    import { buttonVariants } from "$lib/shadcn/components/ui/button";
    import { mutation } from "$lib/utils.svelte";

    let {
        open = $bindable(false),
        group,
        onconfirm
    }: {
        open?: boolean;
        group: { id: string; displayName: string };
        onconfirm: () => Promise<void>;
    } = $props();

    const mutLeave = mutation(onconfirm);
</script>

<Popup alert bind:open>
    {#snippet trigger()}
        <AlertDialog.Trigger class={[buttonVariants({ variant: "destructive" }), "shadow-md"]}>
            Leave Group
        </AlertDialog.Trigger>
    {/snippet}
    <AlertDialog.Header>
        <AlertDialog.Title>Are you sure?</AlertDialog.Title>
        <AlertDialog.Description>
            This will remove you from the group <em>{group.displayName}</em>.
        </AlertDialog.Description>
    </AlertDialog.Header>
    <AlertDialog.Footer>
        <AlertDialog.Cancel class="shadow-md">Cancel</AlertDialog.Cancel>
        <AlertDialog.Action
            class={["shadow-md", mutLeave.applying && "animate-pulse"]}
            disabled={mutLeave.applying}
            onclick={() => (mutLeave.desiredValue = true)}
        >
            Leave
        </AlertDialog.Action>
    </AlertDialog.Footer>
</Popup>
