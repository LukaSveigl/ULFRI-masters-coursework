<script lang="ts">
    import { goto } from "$app/navigation";
    import { page } from "$app/state";
    import Avatar from "$lib/components/Avatar.svelte";
    import InputEur from "$lib/components/InputEur.svelte";
    import Label from "$lib/components/Label.svelte";
    import PopupAddToLoan from "$lib/components/PopupAddToLoan.svelte";
    import { Button } from "$lib/shadcn/components/ui/button";
    import { ScrollArea } from "$lib/shadcn/components/ui/scroll-area";
    import * as Select from "$lib/shadcn/components/ui/select";
    import { sortGroups, sortUsers } from "$lib/sort.js";
    import { mutation } from "$lib/utils.svelte.js";
    import { X } from "lucide-svelte";
    import { SvelteMap } from "svelte/reactivity";
    import { onGetGroupMembers, onGetUsers, onIssueLoan } from "./page.telefunc.js";

    let { data } = $props();

    let idToGroup = $derived(new Map(data.groups.map((x) => [x.id, x])));
    let idToUser = $state(
        new SvelteMap<
            string,
            {
                user: {
                    id: string;
                    displayName: string;
                    avatar: boolean;
                    avatarModified: Date;
                    username: string;
                };
                amount: number | null;
            }
        >()
    );

    let form: HTMLFormElement | null = $state(null);
    let selectedAmount: number | null = $state(null);
    let selectedGroup = $state(page.url.searchParams.get("group") ?? "");
    let selectedUsers: string[] = $state([]);
    let selectedGroupUsers: {
        id: string;
        displayName: string;
        avatar: boolean;
        avatarModified: Date;
        username: string;
    }[] = $state([]);

    const mutFetchGroup = mutation(async (id: string) => {
        const users = (await onGetGroupMembers(id)).toSorted(sortUsers);
        const newIdToUser: typeof idToUser = new SvelteMap();
        const newSelectedUsers: typeof selectedUsers = [];
        for (const user of users) {
            const proxied = $state({ user, amount: null });
            newIdToUser.set(user.id, proxied);
            newSelectedUsers.push(user.id);
        }
        idToUser = newIdToUser;
        selectedUsers = newSelectedUsers;
        selectedGroupUsers = users;
    });

    $effect.pre(() => {
        if (idToGroup.has(selectedGroup)) {
            mutFetchGroup.desiredValue = selectedGroup;
        } else {
            idToUser.clear();
            selectedUsers = [];
            selectedGroupUsers = [];
        }
    });

    const mutLoadUsers = mutation(async (ids: string[]) => {
        const alreadyExist = new Set(selectedUsers);
        const users = (await onGetUsers(ids))
            .filter((x) => !alreadyExist.has(x.id))
            .sort(sortUsers);
        for (const user of users) {
            const proxied = $state({ user, amount: null });
            idToUser.set(user.id, proxied);
            selectedUsers.push(user.id);
        }
    });

    if (page.url.searchParams.has("user")) {
        mutLoadUsers.desiredValue = [page.url.searchParams.get("user")!];
    }

    let finalAmount = $derived(
        selectedAmount != null && selectedAmount > 0 ? selectedAmount : null
    );

    let remainingAmounts = $derived.by(() => {
        if (finalAmount == null) return null;
        if (selectedUsers.length == 0) return null;

        let nAutomatic = 0;
        let manualAmount = 0;
        for (const userId of selectedUsers) {
            const { amount } = idToUser.get(userId)!;
            if (amount == null) {
                nAutomatic++;
            } else {
                manualAmount += amount;
            }
        }

        const remainingAmount = finalAmount - manualAmount;
        const onEachStep = ~~(remainingAmount / nAutomatic);
        const onFirstStep = remainingAmount - nAutomatic * onEachStep;

        const ret = new Map<string, number>();
        let isFirst = true;
        for (const userId of selectedUsers) {
            const { amount } = idToUser.get(userId)!;
            if (amount == null) {
                ret.set(userId, onEachStep + (isFirst ? onFirstStep : 0));
            }
            isFirst = false;
        }

        return ret;
    });

    let finalAmounts = $derived.by(() => {
        if (finalAmount == null) return null;
        if (remainingAmounts == null) return null;

        let ret = new Map<string, number>();
        for (const userId of selectedUsers) {
            const { amount } = idToUser.get(userId)!;
            ret.set(userId, amount == null ? remainingAmounts.get(userId)! : amount);
        }

        return ret;
    });

    let finalAmountsOk = $derived.by(() => {
        if (finalAmounts == null) return false;
        let x = 0;
        for (const amount of finalAmounts.values()) {
            if (amount <= 0) return false;
            x += amount;
        }
        return x == finalAmount;
    });

    let canSubmit = $derived(finalAmount != null && selectedUsers.length > 0 && finalAmountsOk);

    const mutIssueLoan = mutation(
        async (config: {
            groupId: string | null;
            payments: { userId: string; amount: number }[];
        }) => {
            await onIssueLoan(config.groupId, config.payments);
            goto("/history");
        }
    );

    let issueLoanFullErr = $derived(
        mutIssueLoan.stalled == "error" ? mutIssueLoan.stallError : null
    );
</script>

{#snippet groupLine(group: (typeof data.groups)[number])}
    <div class="flex items-center gap-2">
        <Avatar user={group} isGroup size="small" />
        <p class="leading-none">{group.displayName}</p>
    </div>
{/snippet}

<form
    class="flex h-full flex-col items-stretch gap-6"
    bind:this={form}
    onsubmit={(e) => {
        e.preventDefault();
        if (!canSubmit) return;
        mutIssueLoan.desiredValue = {
            groupId: idToGroup.get(selectedGroup)?.id ?? null,
            payments: selectedUsers.map((x) => ({ userId: x, amount: finalAmounts!.get(x)! }))
        };
    }}
>
    <ScrollArea
        orientation="vertical"
        class="relative flex h-full flex-col items-stretch rounded-md border-2 border-neutral-300"
    >
        <div class="m-4 flex flex-col items-stretch gap-4">
            <Label label="Amount">
                <InputEur
                    min="0.01"
                    required
                    disabled={mutIssueLoan.applying}
                    bind:value={selectedAmount}
                />
            </Label>
            <div
                class="grid grid-cols-[1fr_auto] grid-rows-[auto_auto] items-stretch justify-items-stretch gap-x-4 gap-y-[6px]"
            >
                <Label class="row-span-2 grid grid-rows-subgrid" label="Group">
                    <Select.Root
                        type="single"
                        disabled={mutIssueLoan.applying}
                        bind:value={selectedGroup}
                    >
                        <Select.Trigger>
                            {@const group = idToGroup.get(selectedGroup)}
                            {#if group}
                                {@render groupLine(group)}
                            {:else}
                                None
                            {/if}
                        </Select.Trigger>
                        {#if data.groups.length}
                            <Select.Content>
                                {#each data.groups.toSorted(sortGroups) as group (group.id)}
                                    <Select.Item value={group.id}>
                                        {@render groupLine(group)}
                                    </Select.Item>
                                {/each}
                            </Select.Content>
                        {/if}
                    </Select.Root>
                </Label>
                <Label class="row-span-2 grid grid-rows-subgrid" label="Extra Users">
                    <PopupAddToLoan
                        disabled={mutIssueLoan.applying}
                        extraUsers={(() => {
                            const group = idToGroup.get(selectedGroup);
                            if (group == null) return [];
                            return selectedGroupUsers;
                        })()}
                        presentIds={new Set(selectedUsers)}
                        onconfirm={(x) => {
                            mutLoadUsers.desiredValue = x;
                        }}
                    />
                </Label>
            </div>
            <ul class="grid grid-cols-[auto_1fr_auto]">
                {#each selectedUsers as userId (userId)}
                    {@const entry = idToUser.get(userId)!}
                    <li
                        class={[
                            "col-span-3 grid grid-cols-subgrid items-center gap-4 border-t border-inherit py-2 last:border-b",
                            mutIssueLoan.applying && "cursor-not-allowed"
                        ]}
                    >
                        <div
                            class={[
                                "flex items-center gap-4",
                                mutIssueLoan.applying && "opacity-50"
                            ]}
                        >
                            <Avatar user={entry.user} />
                            <p>{entry.user.displayName}</p>
                        </div>
                        <InputEur
                            class="justify-self-end"
                            min="0.01"
                            required
                            disabled={mutIssueLoan.applying}
                            bind:value={() => {
                                const x = finalAmounts?.get(userId);
                                if (x == 0) return null;
                                return x;
                            },
                            (x) => {
                                if (x == null) {
                                    x = 0;
                                } else if (x == 0) {
                                    x = null;
                                }
                                entry.amount = x;
                            }}
                        />
                        <Button
                            variant="outline"
                            size="icon"
                            disabled={mutIssueLoan.applying}
                            onclick={() => {
                                idToUser.delete(userId);
                                selectedUsers = selectedUsers.filter((x) => x != userId);
                            }}
                        >
                            <X />
                        </Button>
                    </li>
                {/each}
            </ul>
        </div>
    </ScrollArea>
    {#if issueLoanFullErr != null}
        <p class="text-sm text-destructive">{issueLoanFullErr}</p>
    {/if}
    <Button
        class={["shadow-md", mutIssueLoan.applying && "animate-pulse"]}
        type="submit"
        disabled={!canSubmit && !mutIssueLoan.applying}
    >
        Issue Loan
    </Button>
</form>
